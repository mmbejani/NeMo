from typing import List

import torch

import editdistance

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.utils import logging


class Seq2SeqDecoder:

    def __init__(self, tokenizer: TokenizerSpec) -> None:
        self.tokenizer = tokenizer

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = self.tokenizer.ids_to_text(tokens)
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = self.tokenizer.ids_to_tokens(tokens)
        return token_list