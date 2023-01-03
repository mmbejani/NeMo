from typing import List, Tuple

import torch
from torch.nn.functional import log_softmax
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.losses.seq2seq import Seq2SeqLoss
from nemo.collections.asr.metrics.seq2seq import Seq2SeqDecoder
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.core.classes import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.common import typecheck
from nemo.utils import logging, model_utils

from pytorch_lightning import Trainer

__all__ = ["Seq2SeqModel", "Seq2SeqModelWithLM"]

class Seq2SeqModel(ASRModel, ASRBPEMixin, Exportable):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")
        self._setup_tokenizer()
        vocabulary = self.tokenizer.tokenizer.get_vocab()


        with open_dict(cfg):
            cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        num_classes = cfg.decoder["num_classes"]
        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.decoder["num_classes"] = len(vocabulary)

        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size
        self.preprocessor = Seq2SeqModel.from_config_dict(self._cfg.preprocessor)
        self.encoder = Seq2SeqModel.from_config_dict(self._cfg.encoder)
        self.ctc_linear = Seq2SeqModel.from_config_dict(self._cfg.ctc_linear)
        self.sequence_linear = Seq2SeqModel.from_config_dict(self._cfg.sequence_linear)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

        self.loss = Seq2SeqLoss(num_classes=num_classes)

        super().__init__(cfg=cfg, trainer=trainer)

        self.decoding = Seq2SeqDecoder(tokenizer=self.tokenizer)
        
        self._wer = WERBPE(
            decoding=self.decoding,
            use_cer=self._cfg.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )


    def training_step(self, batch, batch_idx):
        # Reset access registry
        if AccessMixin.is_access_enabled():
            AccessMixin.reset_registry(self)

        signal, signal_len, transcript, transcript_len = batch
        seq_output, ctc_output, encoded_len = self.forward(input_signal=signal, 
                                                           input_signal_length=signal_len, 
                                                           target_length=transcript_len)
        loss_value, seq_loss, ctc_loss = self.loss(log_probs=ctc_output, logits=seq_output, 
                                                   targets=transcript, inputs_len=encoded_len,
                                                   targets_len=transcript_len)

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
                targets=transcript,
                target_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            wer, _, _ = self._wer.compute()
            self._wer.reset()
            tensorboard_logs.update({'training_batch_wer': wer})

        return {'loss': loss_value, 'log': tensorboard_logs}
        

    @typecheck()
    def forward(self, input_signal, input_signal_length=None, target_length=None) -> Tuple[torch.Tensor]:
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

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(processed_signal, processed_signal_length)
        encoded, encoded_len = encoder_output[0], encoder_output[1]
        ctc_prediction = self.ctc_linear(encoded)
        logits_list = []

        bos_tokens = torch.ones(size=[encoded.size(1), 1], dtype=torch.long) * self.bos_id
        output_tensor = self.embedding(bos_tokens)
        for i in range(self.max_seq_len if target_length is None else target_length):
            decoder_output = self.decoder(output_tensor, encoded, encoded_len)
            logits = self.sequence_linear(decoder_output)[:, -1].detach()
            logits_list.append(logits)
            next_tokens = torch.argmax(logits, dim=-1)
            output_tensor = torch.cat([output_tensor, self.embedding(next_tokens)], dim=1)
        
        logits = torch.cat(logits_list, dim=1)

        return logits, ctc_prediction, encoded_len


    @torch.no_grad()
    def transcribe(self, audio_bytes: List[bytes]) -> List[str]:
        pass


class Seq2SeqModelWithLM(Seq2SeqModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        return super().transcribe(paths2audio_files, batch_size)