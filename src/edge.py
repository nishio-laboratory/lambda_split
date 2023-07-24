import copy

import numpy as np
import torch
from transformers import LlamaTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.base import Base


class Edge(Base):
    def __init__(
            self, 
            first_split_layer_indices: set,
            second_split_layer_indices: set
     ) -> None:
        
        super().__init__(first_split_layer_indices, second_split_layer_indices)

        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = 0 # unk

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self.get_largest_first_model()
        self.get_largest_third_model()

    def get_largest_first_model(self) -> None:
        self.first_model = self.load_model(position='first')

        if self.replace_unused_layers_with_identity:
            # [0, max_first_split_layer_index) 以外を ExtendedIdentity で置き換える
            self.first_model.base_model.model.model.replace_unused_layers_with_identity(
                max_first_split_layer_index=self.max_first_split_layer_index
            )

    def get_largest_third_model(self) -> None:
        self.third_model = self.load_model(position='third')

        if self.replace_unused_layers_with_identity:
            # [min_second_split_layer_index, self.num_decoder_layers) 以外を ExtendedIdentity で置き換える
            self.third_model.base_model.model.model.replace_unused_layers_with_identity(
                min_second_split_layer_index=self.min_second_split_layer_index
            )

    def infer_first_model(
            self, 
            input_ids: torch.Tensor
    ) -> torch.HalfTensor:
        split_first_layer_relative_index = self.rng.integers(0, self.num_first_split_layer_indices)
        split_first_layer_index = self.first_split_layer_indices[split_first_layer_relative_index]

        with torch.no_grad():
            first_feature_vector = self.first_model(
                input_ids, 
                split_first_layer_index=split_first_layer_index
            )

        return first_feature_vector

    def infer_third_model(
            self, 
            second_feature_vector: torch.HalfTensor
    ) -> CausalLMOutputWithPast:
        split_second_layer_relative_index = self.rng.integers(0, self.num_second_split_layer_indices)
        split_second_layer_index = self.second_split_layer_indices[split_second_layer_relative_index]

        with torch.no_grad():
            output = self.third_model(
                inputs_embeds=second_feature_vector, 
                split_second_layer_index=split_second_layer_index
            )

        return output