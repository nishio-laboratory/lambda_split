import copy
from typing import List

import numpy as np
import torch

from src.base import Base
from src.util import SplitComputingConfig, LLMConfig, SimplifiedGenerationConfig


class Cloud(Base):
    def __init__(
            self, 
            split_computing_config: SplitComputingConfig,
            llm_config: LLMConfig,
            simplified_generation_config: SimplifiedGenerationConfig
        ) -> None:
        
        super().__init__(split_computing_config, llm_config)

        self.split_computing_config = split_computing_config
        self.simplied_generation_config = simplified_generation_config

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self.second_model = self._get_largest_second_model()

        if self.split_computing_config.use_split_cache:
            # 過去の first_feature_vector を split_layer_index ごとに保存しておく
            self.stored_first_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.num_decoder_layers + 1)]
            for split_layer_index in self.first_split_layer_indices:
                self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = torch.empty((1, 0, self.num_embed_dims), dtype=torch.half).to(self.device)

            # すでに送信した second_feature_vector_with_past の latest_past_index を split_layer_index ごとに保存しておく
            self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.num_decoder_layers + 1)]
            for split_layer_index in self.second_split_layer_indices:
                self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = 0


    def _get_largest_second_model(self):
        second_model = self.load_model(position='second')

        if self.do_replace_unused_layers_with_identity:
            # [min_first_split_layer_index, max_second_split_layer_index) 以外を ExtendedIdentity で置き換える
            second_model.base_model.model.model.replace_unused_layers_with_identity(
                min_first_split_layer_index=self.min_first_split_layer_index,
                max_second_split_layer_index=self.max_second_split_layer_index
            )

        return second_model
    
    def infer_second_model(
        self, 
        first_feature_vector_for_send: torch.Tensor,
        split_first_layer_index: int,
        split_second_layer_index: int
    ) -> torch.Tensor:
        first_feature_vector_for_send = first_feature_vector_for_send.to(self.device).half()

        if self.split_computing_config.use_split_cache:
            self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_first_layer_index] = torch.cat((
                self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_first_layer_index], 
                first_feature_vector_for_send), 
                dim=1
            )
            first_feature_vector = self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_first_layer_index]
        else:
            first_feature_vector = first_feature_vector_for_send

        with torch.no_grad():
            second_feature_vector = self.second_model(
                inputs_embeds=first_feature_vector, 
                split_first_layer_index=split_first_layer_index, 
                split_second_layer_index=split_second_layer_index,
                use_cache=self.simplied_generation_config.use_past_cache
            )

        if self.split_computing_config.use_split_cache:
            second_feature_vector_for_send = self._delete_already_sent_second_feature_vector_indices(
                second_feature_vector=second_feature_vector,
                split_second_layer_index=split_second_layer_index
            )
        else:
            second_feature_vector_for_send = second_feature_vector

        if self.split_computing_config.dropout_rate < 1.0:
            second_feature_vector_for_send = torch.nn.functional.dropout(
                input=second_feature_vector_for_send, 
                p=1.0 - self.split_computing_config.dropout_rate, 
                training=False
            )

        if self.split_computing_config.quantize_method is not None:
            pass # TODO

        return second_feature_vector_for_send
    
    def _delete_already_sent_second_feature_vector_indices(
            self,
            second_feature_vector: torch.Tensor,
            split_second_layer_index: int
    ) -> torch.Tensor:
        # まだ送信していない past_index のみを取り出す
        sent_latest_past_index_of_second_feature_vector_with_past = self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_second_layer_index]
        second_feature_vector_for_send = second_feature_vector[:, sent_latest_past_index_of_second_feature_vector_with_past:, :]

        # latest_past_index の更新
        self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_second_layer_index] = second_feature_vector.shape[1]
        
        return second_feature_vector_for_send