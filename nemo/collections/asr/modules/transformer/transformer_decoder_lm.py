# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch import nn

from nemo.collections.asr.modules.transformer.transformer_modules import PositionWiseFF

__all__ = ["TransformerDecoderLM"]

class TransformerDecoderBlockLM(nn.Module):

    def __init__(self,
                hidden_size: int,
                inner_size: int,
                num_attention_heads: int = 1,
                attn_layer_dropout: float = 0.0,
                ffn_dropout: float = 0.0,
                hidden_act: str = "gelu",
                pre_ln: bool = False) -> None:
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=hidden_size,
                                         num_heads=num_attention_heads,
                                         dropout=attn_layer_dropout)
        self.pos_ffn = PositionWiseFF(hidden_size=hidden_size, 
                                      inner_size=inner_size,
                                      ffn_dropout=ffn_dropout,
                                      hidden_act=hidden_act)
        
        self.norm1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.norm2 = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, decoder_query, decoder_mask, decoder_keys, encoder_states, encoder_mask):
        pass

class TransformerDecoderLM(nn.Module):

    def __init__(self,
                num_layers: int,
                hidden_size: int,
                inner_size: int,
                num_attention_heads: int = 1,
                attn_layer_dropout: float = 0.0,
                ffn_dropout: float = 0.0,
                hidden_act: str = "gelu",
                pre_ln: bool = False) -> None:
        super().__init__()
