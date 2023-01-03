from typing import List

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

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
        loss_value = self.loss()
        
        




    @typecheck()
    def forward(self, input_signal, input_signal_length=None, target_length=None):
        encoder_output = self.encoder(input_signal, input_signal_length)
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

        return logits_list, ctc_prediction, encoded_len


    @torch.no_grad()
    def transcribe(self, audio_bytes: List[bytes]) -> List[str]:
        pass






class Seq2SeqModelWithLM(Seq2SeqModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        return super().transcribe(paths2audio_files, batch_size)