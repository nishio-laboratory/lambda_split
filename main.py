import io
import time

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

from src.cloud import Cloud
from src.edge import Edge
from src.util import Prompter, SplitComputingConfig, LLMConfig, SimplifiedGenerationConfig


def main(first_split_layer_indices, second_split_layer_indices, random_seed):
    # Edge での SplitComputingConfig
    split_computing_config_edge = SplitComputingConfig(
        device='cpu',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_cache=True,
    )

    # Cloud での SplitComputingConfig
    split_computing_config_cloud = SplitComputingConfig(
        device='cuda',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_cache=True,
    )

    # LLM の Config
    llm_config = LLMConfig(
        base_model='huggyllama/llama-7b',
        lora_weights="tloen/alpaca-lora-7b"
    )

    # テキスト生成の Config
    simplified_generation_config = SimplifiedGenerationConfig(
        max_new_tokens=100,
        use_past_cache=True
    )
    

    # Edge と Cloud のインスタンス
    edge = Edge(split_computing_config_edge, llm_config, simplified_generation_config)
    cloud = Cloud(split_computing_config_cloud, llm_config, simplified_generation_config)

    # 乱数生成器
    rng = np.random.default_rng(random_seed)

    instruction = "Tell me about Japan."
    input = None
    prompter = Prompter('')
    prompt = prompter.generate_prompt(instruction, input)

    inputs = edge.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    start = time.time()

    for idx in tqdm(range(simplified_generation_config.max_new_tokens)):
        print(idx)

        # Triadic split computing : edge -> cloud -> edge
        ## 分割するレイヤの箇所を乱数で決める
        ### first_split_layer
        split_first_layer_relative_index = rng.integers(0, edge.num_first_split_layer_indices)
        split_first_layer_index = edge.first_split_layer_indices[split_first_layer_relative_index]
        ### second_split_layer
        split_second_layer_relative_index = rng.integers(0, edge.num_second_split_layer_indices)
        split_second_layer_index = edge.second_split_layer_indices[split_second_layer_relative_index]

        ## First model : 0 から split_first_layer_index の層まで推論する
        first_feature_vector_for_send = edge.infer_first_model(input_ids, split_first_layer_index)

        ## Second model : split_first_layer_index から split_second_layer_index の層まで推論する
        second_feature_vector_for_send = cloud.infer_second_model(first_feature_vector_for_send, split_first_layer_index, split_second_layer_index)
        
        ## Third model : split_second_layer_index から 最後の層 (self.num_decoder_layers) まで推論する
        output = edge.infer_third_model(second_feature_vector_for_send, split_second_layer_index)

        ## 推論結果のロジットから次のトークンを選択する
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, descending=True, dim=-1)
        
        if sorted_ids[0] == edge.tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)

        '''
        with open(f'torchinfo_summary_log/first_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
            f.write(repr(torchinfo.summary(edge.first_model, input_data=input_ids, depth=10, col_width=50)))
        with open(f'torchinfo_summary_log/second_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
            f.write(repr(torchinfo.summary(cloud.second_model, input_data=first_feature_vector, depth=10, col_width=50)))
        with open(f'torchinfo_summary_log/third_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
            f.write(repr(torchinfo.summary(edge.third_model, input_data=second_feature_vector, depth=10, col_width=50)))
        '''

    print(edge.tokenizer.decode(input_ids[0]))
    print()

    end = time.time()
    print(f'Inference time : {end - start} seconds')

    if split_computing_config_edge.measure_tensor_size_method is not None:
        total_bit_send = sum(edge.send_tensor_size_list) / (1024 ** 2)
        total_bit_receive = sum(edge.receive_tensor_size_list) / (1024 ** 2)
        print(f'{total_bit_send=} Mbit, {total_bit_receive=} Mbit')
        print()

        print(f'{edge.send_tensor_size_list=}')
        print(f'{edge.receive_tensor_size_list=}')


if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    first_split_layer_indices = [1, 2, 3, 4, 5]
    second_split_layer_indices = [27, 28, 29, 30, 31]

    main(first_split_layer_indices, second_split_layer_indices, 42)