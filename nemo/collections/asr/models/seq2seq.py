import torch

from typing import List

from nemo.core.classes import Exportable
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.core.classes.common import typecheck

from pytorch_lightning import Trainer

from omegaconf import DictConfig

__all__ = ["Seq2SeqModel", "Seq2SeqModelWithLM"]

class Seq2SeqModel(ASRModel, ASRBPEMixin, Exportable):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)


    def training_step(self, batch, batch_idx):
        pass

    @typecheck()
    def forward(self, input_signal, input_signal_length=None):
        encoder_output = self.encoder(input_signal, input_signal_length)
        encoded, encoded_len = encoder_output[0], encoder_output[1]
        logits_list = []

        bos_tokens = torch.ones(size=[encoded.size(1), 1], dtype=torch.long) * self.bos_id
        output_tensor = self.embedding(bos_tokens)
        for i in range(self.max_seq_len if input_signal_length is None else input_signal_length):
            decoder_output = self.decoder(output_tensor, encoded, encoded_len)
            logits = self.linear(decoder_output)[:, -1].detach()
            logits_list.append(logits)
            next_tokens = torch.argmax(logits, dim=-1)
            output_tensor = torch.cat([output_tensor, self.embedding(next_tokens)], dim=1)

        return logits_list


    @torch.no_grad()
    def transcribe(self, audio_bytes: List[bytes]) -> List[str]:
        pass






class Seq2SeqModelWithLM(Seq2SeqModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)
        

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        return super().transcribe(paths2audio_files, batch_size)