import copy
from typing import List

import numpy as np
import torch

from src.base_save_bandwidth import Base
from src.util import SplitComputingConfig, LLMConfig

class Cloud(Base):
    def __init__(
            self, 
            split_computing_config: SplitComputingConfig,
            llm_config: LLMConfig
        ) -> None:
        
        super().__init__(split_computing_config, llm_config)

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self._get_largest_second_model()

        # 過去の first_feature_vector を split_layer_index ごとに保存しておく
        self.stored_first_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.num_decoder_layers + 1)]
        for split_layer_index in self.first_split_layer_indices:
            self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = torch.empty((1, 0, self.num_embed_dims), dtype=torch.half).to(self.device)

        # すでに送信した second_feature_vector_with_past の latest_past_index を split_layer_index ごとに保存しておく
        self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.num_decoder_layers + 1)]
        for split_layer_index in self.second_split_layer_indices:
            self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = 0


    def _get_largest_second_model(self):
        self.second_model = self.load_model(position='second')

        if self.do_replace_unused_layers_with_identity:
            # [min_first_split_layer_index, max_second_split_layer_index) 以外を ExtendedIdentity で置き換える
            self.second_model.base_model.model.model.replace_unused_layers_with_identity(
                min_first_split_layer_index=self.min_first_split_layer_index,
                max_second_split_layer_index=self.max_second_split_layer_index
            )
    
    def infer_second_model(
        self, 
        first_feature_vector_for_send: torch.HalfTensor,
        split_first_layer_index: int,
        split_second_layer_index: int
    ) -> torch.HalfTensor:

        self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_first_layer_index] = torch.cat((
            self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_first_layer_index], 
            first_feature_vector_for_send), 
            dim=1
        )

        with torch.no_grad():
            second_feature_vector = self.second_model(
                inputs_embeds=self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_first_layer_index], 
                split_first_layer_index=split_first_layer_index, 
                split_second_layer_index=split_second_layer_index
            )

        return second_feature_vector
    
    def get_second_feature_vector_for_send(
            self,
            second_feature_vector: torch.HalfTensor,
            split_second_layer_index: int
    ) -> torch.HalfTensor:
        # まだ送信していない past_index のみを取り出す
        sent_latest_past_index_of_second_feature_vector_with_past = self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_second_layer_index]
        second_feature_vector_for_send = second_feature_vector[:, sent_latest_past_index_of_second_feature_vector_with_past:, :]

        # latest_past_index の更新
        self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_second_layer_index] = second_feature_vector.shape[1]
        
        return second_feature_vector_for_send