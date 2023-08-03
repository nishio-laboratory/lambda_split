import io
import time

import numpy as np
import torch
import torchinfo
from tqdm import tqdm
import gradio as gr

from src.cloud import Cloud
from src.edge import Edge
from src.util import Prompter, SplitComputingConfig, LLMConfig, SimplifiedGenerationConfig, select_next_token


def main(first_split_layer_indices, second_split_layer_indices, random_seed, ui):
    # Edge での SplitComputingConfig
    split_computing_config_edge = SplitComputingConfig(
        device='cuda',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=True,
    )

    # Cloud での SplitComputingConfig
    split_computing_config_cloud = SplitComputingConfig(
        device='cuda',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=True,
    )

    # LLM の Config
    llm_config_llama_7b = LLMConfig(
        base_model='huggyllama/llama-7b',
        lora_weights="tloen/alpaca-lora-7b"
    )
    llm_config_llama_13b = LLMConfig(
        base_model='huggyllama/llama-13b',
        lora_weights="Angainor/alpaca-lora-13b"
    )
    llm_config_llama_30b = LLMConfig(
        base_model='huggyllama/llama-30b',
        lora_weights="baseten/alpaca-30b"
    )
    llm_config = llm_config_llama_13b

    # Edge と Cloud のインスタンス
    edge = Edge(split_computing_config_edge, llm_config)
    cloud = Cloud(split_computing_config_cloud, llm_config)

    # テキスト生成の Config
    simplified_generation_config = SimplifiedGenerationConfig(
        max_new_tokens=500,
        do_sample=True,
        use_split_past_cache=False,
        temperature=1,
        top_k=50,
        top_p=0.9
    )

    # 乱数生成器
    rng = np.random.default_rng(random_seed)

    def infer(message, history):
        inference_instruction = message
        inference_input = None
        prompter = Prompter('')
        prompt = prompter.generate_prompt(inference_instruction, inference_input)

        inputs = edge.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(edge.device)

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
            logits = output.logits[0, -1, :]
            next_token = select_next_token(logits, simplified_generation_config)
            
            if next_token == edge.tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            yield prompter.get_response(edge.tokenizer.decode(input_ids[0]))

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

        edge.reset_split_sent_cache()
        cloud.reset_split_sent_cache()


    if ui:
        demo = gr.ChatInterface(
            fn=infer,
            title='Demo : Triadic Split Computing for LLM'
        ).queue()
        demo.launch(ssl_verify=False, server_name='0.0.0.0')

    else:
        while True:
            message = input('Instruction : ') # "Tell me about Japan."
            infer(message, None)
            

if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    first_split_layer_indices = [1, 2, 3, 4, 5]
    second_split_layer_indices = [27, 28, 29, 30, 31]

    main(first_split_layer_indices, second_split_layer_indices, 42, True)