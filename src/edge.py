import copy
from typing import List
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.base import Base
from src.util import SplitComputingConfig, SimplifiedGenerationConfig, measure_tensor_size


class Edge(Base):
    def __init__(
            self, 
            split_computing_config: SplitComputingConfig,
            base_model: str
        ) -> None:
        
        super().__init__(split_computing_config, base_model)

        self.split_computing_config = split_computing_config
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token_id = 0 # unk

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self.first_model = self._get_largest_first_model()
        self.third_model = self._get_largest_third_model()

        if self.split_computing_config.use_split_sent_cache:
            self.reset_split_sent_cache()

    def reset_split_sent_cache(self):
        # すでに送信した first_feature_vector_with_past の latest_past_index を split_layer_index ごとに保存しておく
        self.sent_latest_past_index_of_first_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.num_decoder_layers + 1)]
        for split_layer_index in self.second_split_layer_indices:
            self.sent_latest_past_index_of_first_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = 0

        # 過去の second_feature_vector_with_past を split_layer_index ごとに保存しておく
        self.stored_second_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.num_decoder_layers + 1)]
        for split_layer_index in self.second_split_layer_indices:
            self.stored_second_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = torch.empty((1, 0, self.num_embed_dims), dtype=torch.half).to(self.device)

        # 送受信テンソルのサイズを保存しておく
        if self.split_computing_config.measure_tensor_size_method is not None:
            self.send_tensor_size_list = []
            self.receive_tensor_size_list = []

    def _get_largest_first_model(self) -> None:
        first_model = self.load_model(position='first')

        if self.do_replace_unused_layers_with_identity:
            # [0, max_first_split_layer_index) 以外を ExtendedIdentity で置き換える
            first_model.replace_unused_layers_with_identity()
            first_model.model.replace_unused_layers_with_identity(
                max_first_split_layer_index=self.max_first_split_layer_index
            )
            torch.cuda.empty_cache()

        return first_model

    def _get_largest_third_model(self) -> None:
        third_model = self.load_model(position='third')

        if self.do_replace_unused_layers_with_identity:
            # [min_second_split_layer_index, self.num_decoder_layers) 以外を ExtendedIdentity で置き換える
            third_model.replace_unused_layers_with_identity()
            third_model.model.replace_unused_layers_with_identity(
                min_second_split_layer_index=self.min_second_split_layer_index
            )
            torch.cuda.empty_cache()
        
        return third_model

    def infer_first_model(
            self, 
            input_ids: torch.LongTensor,
            first_split_layer_index: int
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device) #.long()

        # first_model の推論
        with torch.no_grad():
            first_feature_vector = self.first_model(
                input_ids=input_ids,
                first_split_layer_index=first_split_layer_index
            )

        if self.split_computing_config.use_split_sent_cache:
            # すでに送信した past_index を削除する
            first_feature_vector_for_send = self._delete_already_sent_first_feature_vector_indices(
                first_feature_vector=first_feature_vector,
                first_split_layer_index=first_split_layer_index
            )
        else:
            first_feature_vector_for_send = first_feature_vector

        # ドロップアウト
        if self.split_computing_config.dropout_rate < 1.0:
            first_feature_vector_for_send = torch.nn.functional.dropout(
                first_feature_vector_for_send, 
                p=self.split_computing_config.dropout_rate, 
                training=False
            )

        # 量子化
        if self.split_computing_config.quantize_method is not None:
            raise NotImplementedError

        # テンソルサイズ (bit) を計測
        if self.split_computing_config.measure_tensor_size_method is not None:
            print(f"{first_feature_vector_for_send.shape}")
            send_tensor_size = measure_tensor_size(
                tensor=first_feature_vector_for_send,
                method=self.split_computing_config.measure_tensor_size_method
            )
            self.send_tensor_size_list.append(send_tensor_size)

        # テンソルサイズに応じて通信レイテンシを設定
        if self.split_computing_config.wait_latency:
            wait_time = send_tensor_size / self.split_computing_config.bandwidth
            print(f"Communication latency : {wait_time} seconds")
            time.sleep(wait_time)

        return first_feature_vector_for_send
    
    def _delete_already_sent_first_feature_vector_indices(
            self,
            first_feature_vector: torch.Tensor,
            first_split_layer_index: int
    ) -> torch.Tensor:
        # まだ送信していない past_index のみを取り出す
        sent_latest_past_index_of_first_feature_vector_with_past = self.sent_latest_past_index_of_first_feature_vector_with_past_for_each_split_layer_index[first_split_layer_index]
        first_feature_vector_for_send = first_feature_vector[:, sent_latest_past_index_of_first_feature_vector_with_past:, :]

        # latest_past_index の更新
        self.sent_latest_past_index_of_first_feature_vector_with_past_for_each_split_layer_index[first_split_layer_index] = first_feature_vector.shape[1]
        
        return first_feature_vector_for_send

    def infer_third_model(
            self, 
            second_feature_vector_for_send: torch.Tensor,
            second_split_layer_index: int
    ) -> CausalLMOutputWithPast:
        second_feature_vector_for_send = second_feature_vector_for_send.to(self.device) #.float()

        if self.split_computing_config.measure_tensor_size_method is not None:
            print(f"{second_feature_vector_for_send.shape}")
            receive_tensor_size = measure_tensor_size(
                tensor=second_feature_vector_for_send,
                method=self.split_computing_config.measure_tensor_size_method
            )
            self.receive_tensor_size_list.append(receive_tensor_size)

        if self.split_computing_config.wait_latency:
            wait_time = receive_tensor_size / self.split_computing_config.bandwidth
            print(f"Communication latency : {wait_time} seconds")
            time.sleep(wait_time)

        if self.split_computing_config.use_split_sent_cache:
            self.stored_second_feature_vector_with_past_for_each_split_layer_index[second_split_layer_index] = torch.cat((
                self.stored_second_feature_vector_with_past_for_each_split_layer_index[second_split_layer_index], 
                second_feature_vector_for_send), 
                dim=1
            )
            second_feature_vector = self.stored_second_feature_vector_with_past_for_each_split_layer_index[second_split_layer_index]
        else:
            second_feature_vector = second_feature_vector_for_send

        with torch.no_grad():
            output = self.third_model(
                inputs_embeds=second_feature_vector, 
                second_split_layer_index=second_split_layer_index
            )

        return output

    # ロジットから次のトークンを選択する
    def select_next_token(
            self,
            logits: torch.Tensor,
            config: SimplifiedGenerationConfig
        ) -> torch.Tensor:
        do_sample = config.do_sample
        temperature = config.temperature
        top_k = config.top_k
        top_p = config.top_p

        next_token_logits = logits[:, -1, :]

        if not do_sample:
            # Greedy decoding
            next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
            return next_tokens
        
        # Apply temperature if given
        next_token_logits = next_token_logits / temperature

        # Apply nucleus (top-p) sampling
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # Apply top-k sampling
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample from the filtered distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)[:, None, 0]

        return next_tokens