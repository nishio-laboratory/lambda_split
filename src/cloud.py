import copy
from typing import List

import numpy as np
import torch

from src.base import Base
from src.utils import SplitComputingConfig, LlmConfig


class Cloud(Base):
    def __init__(
            self, 
            split_computing_config: SplitComputingConfig,
            llm_config: LlmConfig
        ) -> None:
        
        super().__init__(split_computing_config, llm_config)

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self.second_model = self._get_largest_second_model()

        # 毎推論時に呼び出す必要がある初期化処理
        self.init_inference()

    def init_inference(self):
        if self.split_computing_config.use_split_sent_cache:
            # 過去の first_feature_vector を split_layer_index ごとに保存しておく
            self.stored_first_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.llm_config.num_decoder_layers + 1)]
            for split_layer_index in self.first_split_layer_indices:
                self.stored_first_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = torch.empty((1, 0, self.llm_config.num_embed_dims), dtype=torch.half).to(self.device)

            # すでに送信した second_feature_vector_with_past の latest_past_index を split_layer_index ごとに保存しておく
            self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.llm_config.num_decoder_layers + 1)]
            for split_layer_index in self.second_split_layer_indices:
                self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = 0

    def _get_largest_second_model(self):
        second_model = self.load_model(position='second')

        if self.do_replace_unused_layers_with_identity:
            # [min_first_split_layer_index, max_second_split_layer_index) 以外を ExtendedIdentity で置き換える
            if self.llm_config.lora_weights is None:
                second_model.replace_unused_layers_with_identity(
                    min_first_split_layer_index=self.min_first_split_layer_index,
                    max_second_split_layer_index=self.max_second_split_layer_index
                )
            else:
                second_model.base_model.model.replace_unused_layers_with_identity(
                    min_first_split_layer_index=self.min_first_split_layer_index,
                    max_second_split_layer_index=self.max_second_split_layer_index
                )
        
        self.free_memory()

        return second_model
    
    def infer_second_model(
        self, 
        first_feature_vector_for_send: torch.Tensor
    ) -> torch.Tensor:
        first_feature_vector_for_send = first_feature_vector_for_send.to(self.device).half()

        ## 分割するレイヤの箇所を乱数で決める (second_split_layer_index も一度に決めてしまうと、shuffle された first_feature_vector を復元できないので、first_split_layer_index とは別に決める)
        first_split_layer_index = self.rng.choice(self.first_split_layer_indices)

        # shuffle された first_feature_vector を 元に戻す
        if self.split_computing_config.do_shuffle:
            first_feature_vector_for_send = torch.flatten(first_feature_vector_for_send)
            randomized_indices = self.rng.permutation(len(first_feature_vector_for_send)) # シャッフル時のインデックスを再現
            reverse_indices = torch.from_numpy(np.argsort(randomized_indices)) # インデックスを元に戻す
            first_feature_vector_for_send = first_feature_vector_for_send[reverse_indices].reshape(1, -1, self.llm_config.num_embed_dims)
        
        # first_feature_vector を復元
        if self.split_computing_config.use_split_sent_cache:
            self.stored_first_feature_vector_with_past_for_each_split_layer_index[first_split_layer_index] = torch.cat((
                self.stored_first_feature_vector_with_past_for_each_split_layer_index[first_split_layer_index], 
                first_feature_vector_for_send), 
                dim=1
            )
            first_feature_vector = self.stored_first_feature_vector_with_past_for_each_split_layer_index[first_split_layer_index]
        else:
            first_feature_vector = first_feature_vector_for_send            

        ## 分割するレイヤの箇所を乱数で決める
        second_split_layer_index = self.rng.choice(self.second_split_layer_indices)

        with torch.no_grad():
            second_feature_vector = self.second_model(
                inputs_embeds=first_feature_vector, 
                first_split_layer_index=first_split_layer_index, 
                second_split_layer_index=second_split_layer_index
            )

        if self.split_computing_config.use_split_sent_cache:
            second_feature_vector_for_send = self._delete_already_sent_second_feature_vector_indices(
                second_feature_vector=second_feature_vector,
                second_split_layer_index=second_split_layer_index
            )
        else:
            second_feature_vector_for_send = second_feature_vector

        # shuffle
        if self.split_computing_config.do_shuffle:
            # flatten してから shuffle して再度 reshape する
            second_feature_vector_for_send = torch.flatten(second_feature_vector_for_send)
            randomized_indices = self.rng.permutation(len(second_feature_vector_for_send))
            second_feature_vector_for_send = second_feature_vector_for_send[torch.from_numpy(randomized_indices)]
            second_feature_vector_for_send = second_feature_vector_for_send.reshape(1, -1, self.llm_config.num_embed_dims)

        if self.split_computing_config.dropout_rate < 1.0:
            second_feature_vector_for_send = torch.nn.functional.dropout(
                input=second_feature_vector_for_send, 
                p=1.0 - self.split_computing_config.dropout_rate, 
                training=False
            )

        if self.split_computing_config.quantize_method is not None:
            raise NotImplementedError

        return second_feature_vector_for_send
    
    def _delete_already_sent_second_feature_vector_indices(
            self,
            second_feature_vector: torch.Tensor,
            second_split_layer_index: int
    ) -> torch.Tensor:
        # まだ送信していない past_index のみを取り出す
        sent_latest_past_index_of_second_feature_vector_with_past = self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[second_split_layer_index]
        second_feature_vector_for_send = second_feature_vector[:, sent_latest_past_index_of_second_feature_vector_with_past:, :]

        # latest_past_index の更新
        self.sent_latest_past_index_of_second_feature_vector_with_past_for_each_split_layer_index[second_split_layer_index] = second_feature_vector.shape[1]
        
        return second_feature_vector_for_send