import io
from typing import List
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.base import Base
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig


class Edge(Base):
    def __init__(
            self, 
            split_computing_config: SplitComputingConfig,
            llm_config: LlmConfig
        ) -> None:
        
        super().__init__(split_computing_config, llm_config)

        self.split_computing_config = split_computing_config
        self.tokenizer = AutoTokenizer.from_pretrained(llm_config.base_model)
        self.tokenizer.pad_token_id = 0 # unk

        # あらかじめ考えられる中で最大のモデルだけを保存しておくことで、メモリを節約する
        self.first_model = self._get_largest_first_model()
        self.third_model = self._get_largest_third_model()
        
        # 毎推論時に呼び出す必要がある初期化処理
        self.init_inference()

    def init_inference(self):
        if self.split_computing_config.use_split_sent_cache:
        # すでに送信した first_feature_vector_with_past の latest_past_index を split_layer_index ごとに保存しておく
            self.sent_latest_past_index_of_first_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.llm_config.num_decoder_layers + 1)]
            for split_layer_index in self.second_split_layer_indices:
                self.sent_latest_past_index_of_first_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = 0

            # 過去の second_feature_vector_with_past を split_layer_index ごとに保存しておく
            self.stored_second_feature_vector_with_past_for_each_split_layer_index = [None for _ in range(self.llm_config.num_decoder_layers + 1)]
            for split_layer_index in self.second_split_layer_indices:
                self.stored_second_feature_vector_with_past_for_each_split_layer_index[split_layer_index] = torch.empty((1, 0, self.llm_config.num_embed_dims), dtype=self.dtype).to(self.device)

        self.past_key_values = None

        self.free_memory()
        self.fix_seed()

    def _get_largest_first_model(self) -> None:
        first_model = self.load_model(position='first')

        if self.do_replace_unused_layers_with_identity:
            # [0, max_first_split_layer_index) 以外を ExtendedIdentity で置き換える
            if self.llm_config.lora_weights is None:
                first_model.replace_unused_layers_with_identity(
                    max_first_split_layer_index=self.max_first_split_layer_index
                )
            else:
                first_model.base_model.model.replace_unused_layers_with_identity(
                    max_first_split_layer_index=self.max_first_split_layer_index
                )
        
        self.free_memory()

        return first_model

    def _get_largest_third_model(self) -> None:
        third_model = self.load_model(position='third')

        if self.do_replace_unused_layers_with_identity:
            # [min_second_split_layer_index, self.llm_config.num_decoder_layers) 以外を ExtendedIdentity で置き換える
            if self.llm_config.lora_weights is None:
                third_model.replace_unused_layers_with_identity(
                    min_second_split_layer_index=self.min_second_split_layer_index
                )
            else:
                third_model.base_model.model.replace_unused_layers_with_identity(
                    min_second_split_layer_index=self.min_second_split_layer_index
                )
        
        self.free_memory()
        
        return third_model
    
    def _update_past_key_values(self) -> None:
        del self.past_key_values
        self.past_key_values = ()

        for idx in range(self.llm_config.num_decoder_layers):
            if self.first_next_cache[idx] is not None and self.third_next_cache[idx] is not None:
                raise ValueError('first_next_cache and third_next_cache are both not None')
            elif self.first_next_cache[idx] is not None:
                self.past_key_values += (self.first_next_cache[idx],)
            elif self.third_next_cache[idx] is not None:
                self.past_key_values += (self.third_next_cache[idx],)
            else:
                self.past_key_values += (None,)

    def infer_first_model(
            self, 
            input_ids: torch.LongTensor
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device) #.long()

        ## 分割するレイヤの箇所を乱数で決める
        first_split_layer_index = self.rng.choice(self.first_split_layer_indices)

        # first_model の推論
        with torch.no_grad():
            first_model_output = self.first_model(
                input_ids=input_ids,
                first_split_layer_index=first_split_layer_index,
                use_cache=self.split_computing_config.use_past_key_values,
                past_key_values=self.past_key_values
            )
            first_feature_vector = first_model_output.last_hidden_state
            
            if self.split_computing_config.use_past_key_values:
                self.first_next_cache = first_model_output.past_key_values

        if self.split_computing_config.use_split_sent_cache:
            # すでに送信した past_index を削除する
            first_feature_vector_for_send = self._delete_already_sent_first_feature_vector_indices(
                first_feature_vector=first_feature_vector,
                first_split_layer_index=first_split_layer_index
            )
        else:
            first_feature_vector_for_send = first_feature_vector

        # shuffle
        if self.split_computing_config.do_shuffle:
            # flatten してから shuffle して再度 reshape する
            first_feature_vector_for_send = torch.flatten(first_feature_vector_for_send)
            randomized_indices = self.rng.permutation(len(first_feature_vector_for_send))
            print(randomized_indices)
            first_feature_vector_for_send = first_feature_vector_for_send[torch.from_numpy(randomized_indices)]
            first_feature_vector_for_send = first_feature_vector_for_send.reshape(1, -1, self.llm_config.num_embed_dims)

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
            second_feature_vector_for_send: torch.Tensor
    ) -> CausalLMOutputWithPast:
        second_feature_vector_for_send = second_feature_vector_for_send.to(self.device).to(torch.float if self.device == 'cpu' else torch.half)

        second_split_layer_index = self.rng.choice(self.second_split_layer_indices)

        # shuffle された second_feature_vector を 元に戻す
        if self.split_computing_config.do_shuffle:
            second_feature_vector_for_send = torch.flatten(second_feature_vector_for_send)
            randomized_indices = self.rng.permutation(len(second_feature_vector_for_send)) # シャッフル時のインデックスを再現
            reverse_indices = torch.from_numpy(np.argsort(randomized_indices)) # インデックスを元に戻す
            second_feature_vector_for_send = second_feature_vector_for_send[reverse_indices].reshape(1, -1, self.llm_config.num_embed_dims)

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
            third_model_output = self.third_model(
                inputs_embeds=second_feature_vector, 
                second_split_layer_index=second_split_layer_index,
                use_cache=self.split_computing_config.use_past_key_values,
                past_key_values=self.past_key_values
            )

            if self.split_computing_config.use_past_key_values:
                self.third_next_cache = third_model_output.past_key_values
                self._update_past_key_values()

        return third_model_output

    # ロジットから次のトークンを選択する
    def sample_next_token(
            self,
            logits: torch.Tensor,
            config: SimplifiedGenerationConfig
        ) -> torch.Tensor:
        '''
        Reference : https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L488
        '''
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