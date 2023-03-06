from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import soundfile as sf
from typing import Dict, List, Optional, Union
from omegaconf import DictConfig, open_dict
from random import random

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.losses.seq2seq import Seq2SeqLoss
from nemo.collections.asr.metrics.seq2seq import Seq2SeqDecoder
from nemo.collections.asr.metrics.wer_bpe import WERS2S, CTCBPEDecoding
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.nlp.models.language_modeling.transformer_lm_model import \
        (get_transformer,
        get_tokenizer)
from nemo.core.classes import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.common import typecheck
from torch.nn import Conv1d
from nemo.utils import logging, model_utils

from pytorch_lightning import Trainer

__all__ = ["Seq2SeqModel", "Seq2SeqModelWithLM"]

class Seq2SeqModel(ASRModel, ASRBPEMixin, Exportable):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.cfg = model_utils.convert_model_config_to_dict_config(cfg)

        if 'encoder_tokenizer' not in cfg and 'decoder_tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
        self._setup_dual_tokenizer(cfg.encoder_tokenizer, cfg.decoder_tokenizer)

        self.world_size = 1
        self.max_seq_len = self.cfg.get('max_len_seq', 100)
        self.teacher_forcing_ratio = self.cfg.get('teacher_forcing_ratio', 0.)
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = Seq2SeqModel.from_config_dict(self.cfg.preprocessor)
        if 'spec_augmentation' in self.cfg:
            self.spec_augmentation = Seq2SeqModel.from_config_dict(self.cfg.spec_augmentation)
        self.encoder = Seq2SeqModel.from_config_dict(self.cfg.encoder)
        self.decoder = Seq2SeqModel.from_config_dict(self.cfg.decoder)
        self.ctc_linear = Seq2SeqModel.from_config_dict(self.cfg.ctc_linear)
        self.connector_conv = Seq2SeqModel.from_config_dict(self.cfg.connector_conv)
        self.decoder_embedding = Seq2SeqModel.from_config_dict(self.cfg.decoder_embedding)
        self.decoder_embedding.to(self.device)
        self.sequence_linear = Seq2SeqModel.from_config_dict(self.cfg.sequence_linear)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.loss = Seq2SeqLoss()

        self.decoding = Seq2SeqDecoder(tokenizer=self.decoder_tokenizer)
        
        self._wer = WERS2S(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", True),
        )


    def training_step(self, batch, batch_idx):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        signal, signal_len, encoder_transcript, encoder_transcript_len, decoder_transcript, decoder_transcript_len = batch
        seq_output, ctc_output, encoded_len = self.forward(input_signal=signal, 
                                                           input_signal_length=signal_len, 
                                                           target=decoder_transcript,
                                                           target_length=decoder_transcript_len)
        loss_value, seq_loss, ctc_loss = self.loss.forward(log_probs=ctc_output, logits=seq_output, 
                                                   encoder_targets=encoder_transcript, 
                                                   decoder_targets=decoder_transcript, 
                                                   input_lengths=encoded_len,
                                                   encoder_target_lengths=encoder_transcript_len,
                                                   decoder_target_lengths=decoder_transcript_len)

        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        tensorboard_logs = {
            'train_loss': loss_value,
            'train_seq_loss': seq_loss,
            'train_ctc_loss': ctc_loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
        else:
            log_every_n_steps = 1

        if (batch_idx + 1) % log_every_n_steps == 0:
            self._wer.update(
                predictions=torch.nn.functional.softmax(seq_output, dim=-1),
                targets=decoder_transcript,
                target_lengths=decoder_transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}
        
    def validation_step(self, batch, batch_idx):
        signal, signal_len, encoder_transcript, encoder_transcript_len, decoder_transcript, decoder_transcript_len = batch
        seq_output, ctc_output, encoded_len = self.forward(input_signal=signal, 
                                                           input_signal_length=signal_len, 
                                                           target=decoder_transcript,
                                                           target_length=decoder_transcript_len)
        loss_value, _, _ = self.loss.forward(log_probs=ctc_output, logits=seq_output, 
                                                   encoder_targets=encoder_transcript, 
                                                   decoder_targets=decoder_transcript, 
                                                   input_lengths=encoded_len,
                                                   encoder_target_lengths=encoder_transcript_len,
                                                   decoder_target_lengths=decoder_transcript_len)


        self._wer.update(
            predictions=torch.nn.functional.softmax(seq_output, dim=-1),
            targets=decoder_transcript,
            target_lengths=decoder_transcript_len,
            predictions_lengths=encoded_len,
        )
        wer, wer_num, wer_denom = self._wer.compute()
        self._wer.reset()
        tensorboard_logs ={'val_loss': loss_value,
            'val_wer_num': wer_num,
            'val_wer_denom': wer_denom,
            'val_wer': wer
        }

        return tensorboard_logs

    @typecheck()
    def forward(self, input_signal, input_signal_length=None, target=None, target_length=None) -> Tuple[torch.Tensor]:
        """Forwad through seq2seq model

        Args:
            input_signal (_type_): _description_
            input_signal_length (_type_, optional): _description_. Defaults to None.
            target_length (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[torch.Tensor]: There are three tensors as following:
                1- The logits output of decoder (unnormlized)
                2- The prediction of encoder based on CTC
                3- The length of output that encoder is processed
        """
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )

        if hasattr(self, "spec_augmentation") and self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)
         
        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = encoder_output[0], encoder_output[1]
        conv_encoded = self.connector_conv(encoded)
        encoded = encoded.transpose(1,2)
        ctc_prediction = self.ctc_linear(encoded)
        ctc_prediction = self.log_softmax(ctc_prediction)

        encoder_mask = self._mask_generator(encoded_len)

        batch_size = encoded.size(0)
        bos_tokens = torch.ones(size=[batch_size, 1], dtype=torch.long).to(self.device) \
                * self.decoder_tokenizer.bos_id

        if target is not None and random() < self.teacher_forcing_ratio:
            decoder_mask = self._mask_generator(target_length)
            bos_target_removed_eos = torch.cat([bos_tokens, target[:, :-1]], dim=1)
            embedding = self.decoder_embedding(bos_target_removed_eos)
            decoder_output = self.decoder(embedding, decoder_mask, conv_encoded, encoder_mask)
            logits = self.sequence_linear(decoder_output)
                    
        else:
            logits_list = []
            if not self.training:
                eos_tokens = [False] * batch_size
            step_length = torch.ones(batch_size, dtype=torch.long)
            output_tensor = self.decoder_embedding(bos_tokens)
            for i in range(self.max_seq_len if target_length is None else torch.max(target_length).item()):
                decoder_mask = self._mask_generator(step_length)
                decoder_output = self.decoder(output_tensor, decoder_mask, encoded, encoder_mask)
                logits = self.sequence_linear(decoder_output)[:, -1]
                logits_list.append(logits.unsqueeze(1))
                next_tokens = torch.argmax(logits.detach(), dim=-1)
                output_tensor = torch.cat([output_tensor, self.decoder_embedding(next_tokens).unsqueeze(1)], dim=1)
                if not self.training:
                    for i in range(batch_size):
                        if next_tokens[i].item() in {self.decoder_tokenizer.eos_id, self.decoder_tokenizer.pad_id}:
                            eos_tokens[i] = True
                    if all(eos_tokens):
                        break
                if i == self.max_seq_len - 1:
                    logging.warning(f'The length of generated sequences exceeds the maximum length of {self.max_seq_len}')
                elif self.training:
                    for i in range(batch_size):
                        if step_length[i] < target_length[i]:
                            step_length[i] += 1
            logits = torch.cat(logits_list, dim=1)
        return logits, ctc_prediction, encoded_len


    @torch.no_grad()
    def transcribe(self, audio_bytes: List[bytes]) -> List[str]:
        audios = [torch.tensor(sf.read(audio_byte)[0], dtype=torch.float32) for audio_byte in audio_bytes]
        audios_length = [audio.size(0) for audio in audios]
        
        input_length = torch.tensor(audios_length).long()
        input_tensor = pad_sequence(audios, batch_first=True)
        
        logits, _, _ = self.forward(input_tensor, input_length)
        tokens = torch.argmax(logits, dim=-1)
    
        transcrptions = [self.decoding.decode_tokens_to_str(token.detach().cpu().numpy().tolist())
                            for token in tokens]
        return transcrptions
        
        

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='labels')

        shuffle = config['shuffle']
        
        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None
            
        dataset = audio_to_text_dataset.get_dual_bpe_dataset(config=config, augmentor=augmentor, encoder_tokenizer=self.encoder_tokenizer, decoder_tokenizer=self.decoder_tokenizer)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

        
    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _mask_generator(self, batch_lengths: torch.Tensor) -> torch.Tensor:
        """Generate mask respect to the given batch_lengths
        B : batch size

        Args:
            batch_lengths (torch.Tensor): lengths of each batch with shape [B]

        Returns:
            torch.Tensor: generated mask
        """
        max_len = torch.max(batch_lengths)
        mask = torch.zeros(batch_lengths.size(0), max_len).long()
        for i in range(batch_lengths.size(0)):
            mask[i,:batch_lengths[i]] = torch.ones(batch_lengths[i]).long()
        return mask.to(self.device)





class Seq2SeqModelWithLM(Seq2SeqModel):

    def __init__(self, cfg: DictConfig):
        """This class is a Module which is not trainable. Therefore, it should be loaded from 
        checkpoint and can not be trained.

        Args:
            cfg (DictConfig): a yaml based configuration which has keys named 
                                - acoustic_model_path
                                - lm_model_path
                              If this value is None then, an exception is raised.
        """
        super().__init__(cfg, None)
        self.cfg = model_utils.convert_model_config_to_dict_config(cfg)
        self._setup_dual_tokenizer(cfg.encoder_tokenizer, cfg.decoder_tokenizer)
        self.beam_size = cfg.get('beam_size',1)

        state_dict = torch.load(cfg.model_path)
        self.load_state_dict(state_dict)

        self.lm = get_transformer(library='huggingface', model_name=cfg.lm_model_name, pretrained=True)
        self.lm.eval()
        self.eval()

        if 'encoder_tokenizer' not in cfg and 'decoder_tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
        self._setup_dual_tokenizer(cfg.encoder_tokenizer, cfg.decoder_tokenizer)

        self.world_size = 1
        self.max_seq_len = self.cfg.get('max_len_seq', 100)
        self.greedy_decoder = CTCBPEDecoding()

        self.preprocessor = Seq2SeqModel.from_config_dict(self.cfg.preprocessor)
        self.encoder = Seq2SeqModel.from_config_dict(self.cfg.encoder)
        self.decoder = Seq2SeqModel.from_config_dict(self.cfg.decoder)
        self.ctc_linear = Seq2SeqModel.from_config_dict(self.cfg.ctc_linear)
        self.decoder_embedding = Seq2SeqModel.from_config_dict(self.cfg.decoder_embedding)
        self.decoder_embedding.to(self.device)
        self.sequence_linear = Seq2SeqModel.from_config_dict(self.cfg.sequence_linear)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        

    def _encoder_transcription(self, input_signal: torch.Tensor,
                               input_signal_length: torch.Tensor) -> Tuple[
                                                                        List[str], 
                                                                        torch.Tensor,
                                                                        torch.Tensor]:
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded, encoded_len = encoder_output[0], encoder_output[1]
        encoded = encoded.transpose(1,2)
        ctc_prediction = self.ctc_linear(encoded)
        ctc_prediction = self.log_softmax(ctc_prediction)
        tokens = torch.argmax(ctc_prediction, dim=-1).detach().cpu().numpy().tolist()

        transcrptions = [self.decoding.decode_tokens_to_str(token)
                            for token in tokens]
        
        return  transcrptions, encoded, encoded_len


        

    def transcribe(self, paths2audio_files: List[str]) -> List[str]:
        """This method transcribes a batch of utterances rely on acoustic model and language model.

        Args:
            paths2audio_files (List[str]): List of path to utterance (sample rate must be 16kHz)
        """
        audio_tensors = [torchaudio.load(path2audio)[0] for path2audio in paths2audio_files]
        pass

    def transcribe_single_file(self, path2audio_file: str) -> str:
        audio_tensosr = torchaudio.load(path2audio_file)[0]