import io
import time

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from dataclasses import asdict

from src.cloud import Cloud
from src.edge import Edge
from src.utils import SplitComputingConfig, SimplifiedGenerationConfig, Prompter, export_split_model_torchinfo_summary


def main(first_split_layer_indices, second_split_layer_indices, random_seed, show_ui):
    # Edge での SplitComputingConfig
    edge_split_computing_config = SplitComputingConfig(
        device='cuda',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=True,
    )

    # Cloud での SplitComputingConfig
    cloud_split_computing_config = SplitComputingConfig(
        device='cuda',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=True,
    )

    # LLM の Config
    ## LLaMa 2 : https://huggingface.co/meta-llama
    base_model_list_llama2 = [ 
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-70b-chat-hf',
    ]
    ## LLaMa : https://huggingface.co/decapoda-research
    base_model_list_llama = [
        'decapoda-research/llama-7b-hf',
        'decapoda-research/llama-13b-hf',
        'decapoda-research/llama-30b-hf',
        'decapoda-research/llama-65b-hf'
    ]
    base_model = base_model_list_llama2[0]

    # Edge と Cloud のインスタンス
    edge = Edge(edge_split_computing_config, base_model)
    cloud = Cloud(cloud_split_computing_config, base_model)

    # Export torchinfo summary
    export_split_model_torchinfo_summary(base_model, edge, cloud, export_dir='torchinfo_summary_log')

    # 乱数生成器
    rng = np.random.default_rng(random_seed)

    def infer(message, history, max_new_tokens, do_sample, temperature, top_k, top_p):
        # テキスト生成の Config
        simplified_generation_config = SimplifiedGenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_split_past_cache=False,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

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
            first_split_layer_relative_index = rng.integers(0, edge.num_first_split_layer_indices)
            first_split_layer_index = edge.first_split_layer_indices[first_split_layer_relative_index]
            ### second_split_layer
            second_split_layer_relative_index = rng.integers(0, edge.num_second_split_layer_indices)
            second_split_layer_index = edge.second_split_layer_indices[second_split_layer_relative_index]

            ## First model : 0 から first_split_layer_index の層まで推論する
            first_feature_vector_for_send = edge.infer_first_model(input_ids, first_split_layer_index)

            ## Second model : first_split_layer_index から second_split_layer_index の層まで推論する
            second_feature_vector_for_send = cloud.infer_second_model(first_feature_vector_for_send, first_split_layer_index, second_split_layer_index)
            
            ## Third model : second_split_layer_index から 最後の層 (self.num_decoder_layers) まで推論する
            output = edge.infer_third_model(second_feature_vector_for_send, second_split_layer_index)

            ## 推論結果のロジットから次のトークンを選択する
            next_tokens = edge.select_next_token(output.logits, simplified_generation_config)
            
            # EOS トークンが生成されたら終了する
            if next_tokens[0, -1] == edge.tokenizer.eos_token_id:
                break
            
            # 次のトークンを input_ids に追加する
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # デトークナイズされたテキストを出力
            cur = time.time()
            yield prompter.get_response(edge.tokenizer.decode(input_ids[0])) + f'\n\n({idx + 1} tokens, {cur - start:.2f} seconds, {idx / (cur - start):.2f} tps)'


        print(edge.tokenizer.decode(input_ids[0]))
        print()

        end = time.time()
        print(f'Inference time : {end - start} seconds')
        print(f'Number of generated tokens : {idx} tokens')
        print(f'Tokens per second : {(end - start) / idx} tps')

        if edge_split_computing_config.measure_tensor_size_method is not None:
            total_bit_send = sum(edge.send_tensor_size_list) / (1024 ** 2)
            total_bit_receive = sum(edge.receive_tensor_size_list) / (1024 ** 2)
            print(f'{total_bit_send=} Mbit, {total_bit_receive=} Mbit')
            print()

            print(f'{edge.send_tensor_size_list=}')
            print(f'{edge.receive_tensor_size_list=}')

        edge.reset_split_sent_cache()
        cloud.reset_split_sent_cache()

        edge.free_memory()
        cloud.free_memory()


    if show_ui:
        with gr.Blocks() as demo:
            gr.Markdown(f"<h1><center>Demo : Triadic Split Computing for {base_model}</center></h1>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"<center>Text Generation Config</center>")

                    max_new_tokens = gr.Slider(minimum=1, maximum=500, value=250, step=1, label="max_new_tokens", interactive=True)
                    do_sample = gr.Checkbox(value=True, label="do_sample", interactive=True)
                    temperature = gr.Slider(minimum=0.1, maximum=5, value=1, label="temperature", interactive=True)
                    top_k = gr.Slider(minimum=1, maximum=1000, value=50, step=1, label="top_k", interactive=True)
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.9, label="top_p", interactive=True)

                with gr.Column(scale=3):
                    gr.ChatInterface(
                        fn=infer,
                        additional_inputs=[max_new_tokens, do_sample, temperature, top_k, top_p]
                    )

        demo.queue().launch(ssl_verify=False, server_name='0.0.0.0')

    else:
        # テキスト生成の Config
        simplified_generation_config = SimplifiedGenerationConfig(
            max_new_tokens=500,
            do_sample=True,
            use_split_past_cache=False,
            temperature=1,
            top_k=50,
            top_p=0.9
        )

        while True:
            message = input('Input text : ')
            for response in infer(message, None, **asdict(simplified_generation_config)):
                print(response)
            

if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    first_split_layer_indices = np.array([1, 2, 3, 4, 5])
    second_split_layer_indices = np.array([-5, -4, -3, -2, -1]) + 32

    main(first_split_layer_indices, second_split_layer_indices, 42, True)