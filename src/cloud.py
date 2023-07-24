import copy

import torch

from src.base import Base


class Cloud(Base):
    def __init__(
            self, 
            first_split_layer_indices: set,
            second_split_layer_indices: set
        ) -> None:
        
        super().__init__(first_split_layer_indices, second_split_layer_indices)

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self.get_largest_second_model()

    def get_largest_second_model(self):
        self.second_model = self.load_model(position='second')

        if self.replace_unused_layers_with_identity:
            # [min_first_split_layer_index, max_second_split_layer_index) 以外を ExtendedIdentity で置き換える
            self.second_model.base_model.model.model.replace_unused_layers_with_identity(
                min_first_split_layer_index=self.min_first_split_layer_index,
                max_second_split_layer_index=self.max_second_split_layer_index
            )
    
    def infer_second_model(
        self, 
        first_feature_vector: torch.HalfTensor
    ) -> torch.HalfTensor:
        split_first_layer_relative_index = self.rng.integers(0, self.num_first_split_layer_indices)
        split_first_layer_index = self.first_split_layer_indices[split_first_layer_relative_index]

        split_second_layer_relative_index = self.rng.integers(0, self.num_second_split_layer_indices)
        split_second_layer_index = self.second_split_layer_indices[split_second_layer_relative_index]

        with torch.no_grad():
            second_feature_vector = self.second_model(
                inputs_embeds=first_feature_vector, 
                split_first_layer_index=split_first_layer_index, 
                split_second_layer_index=split_second_layer_index
            )

        return second_feature_vector