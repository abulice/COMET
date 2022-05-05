# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
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
r"""
XLM-RoBERTa Encoder
==============
    Pretrained XLM-RoBERTa  encoder from Hugging Face.
"""
from typing import Dict, List

import torch
from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
from transformers import XLMRobertaModel, XLMRobertaTokenizer

from math import ceil


class XLMREncoder(BERTEncoder):
    """XLM-RoBERTA Encoder encoder.

    :param pretrained_model: Pretrained model from hugging face.
    """

    def __init__(self, pretrained_model: str) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_model)
        self.model = XLMRobertaModel.from_pretrained(
            pretrained_model, add_pooling_layer=False
        )
        self.model.encoder.output_hidden_states = True

    @classmethod
    def from_pretrained(cls, pretrained_model: str) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.
        :param pretrained_model: Name of the pretrain model to be loaded.

        :return: Encoder model
        """
        return XLMREncoder(pretrained_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        try:
            last_hidden_states, _, all_layers = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=False,
            )
        except:
            breakpoint()
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

    def concat_sequences(self, inputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        concat_input_ids = []
        # Remove padding before concatenation
        for encoder_input in inputs:
            input_ids = encoder_input["input_ids"]
            input_ids = [x.masked_select(x.ne(self.tokenizer.pad_token_id)).tolist() for x in input_ids.unbind(dim=0)]
            concat_input_ids.append(input_ids)

        # Concatenate inputs into a single batch
        batch_size = len(concat_input_ids[0])
        batch = []
        for i in range(batch_size):
            lengths = tuple(len(x[i]) for x in concat_input_ids)
            offset = 1
            
            # self.max_positions = 512 but we need to remove 6 aditional tokens
            # <s>...</s></s>...</s></s>...</s></s>
            special_tokens = 1 + len(inputs)*2
            if sum(lengths) > self.max_positions-special_tokens:
                offset = ceil((sum(lengths) - (self.max_positions-special_tokens)) / len(inputs))
                
            new_sequence = concat_input_ids[0][i]
            for j in range(1, len(inputs)):
                new_sequence = self.tokenizer.build_inputs_with_special_tokens(
                    new_sequence[1:-offset], concat_input_ids[j][i][1:-offset]
                )
            batch.append(torch.tensor(new_sequence))
        
        lengths = [t.shape[0] for t in batch]
        max_len = max(lengths)
        padded = [self.pad_tensor(t, max_len, self.tokenizer.pad_token_id) for t in batch]
        lengths = torch.tensor(lengths, dtype=torch.long)
        padded = torch.stack(padded, dim=0).contiguous()
        attention_mask = torch.arange(max_len)[None, :] < lengths[:, None]
        return {
            "input_ids": padded,
            "attention_mask": attention_mask,
        }
