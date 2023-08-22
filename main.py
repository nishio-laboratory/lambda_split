import time

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from dataclasses import asdict

from src.cloud import Cloud
from src.edge import Edge
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig, SplitComputingLogger, Prompter


def main(
        edge_split_computing_config: SplitComputingConfig,
        cloud_split_computing_config: SplitComputingConfig,
        llm_config: LlmConfig,
        random_seed: int,
        show_ui: bool
    ):

    # Edge と Cloud のインスタンス
    edge = Edge(edge_split_computing_config, llm_config)
    cloud = Cloud(cloud_split_computing_config, llm_config)

    # 乱数生成器
    rng = np.random.default_rng(random_seed)


    def infer_each_request(message, history, max_new_tokens, do_sample, temperature, top_k, top_p, **kwargs):
        # 毎推論時に呼び出す必要がある初期化処理
        edge.init_inference()
        cloud.init_inference()

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

        split_computing_logger = SplitComputingLogger(
            edge_split_computing_config, 
            cloud_split_computing_config, 
            llm_config, 
            simplified_generation_config
        )

        for idx in tqdm(range(simplified_generation_config.max_new_tokens)):
            print(idx)

            # Triadic split computing : edge -> cloud -> edge
            ## 分割するレイヤの箇所を乱数で決める
            first_split_layer_index = rng.choice(edge.first_split_layer_indices)
            second_split_layer_index = rng.choice(edge.second_split_layer_indices)

            inference_start_time = time.perf_counter()

            ## First model : 0 から first_split_layer_index の層まで推論する
            first_feature_vector_for_send = edge.infer_first_model(input_ids, first_split_layer_index)
            first_model_inference_time = time.perf_counter()

            ## Second model : first_split_layer_index から second_split_layer_index の層まで推論する
            second_feature_vector_for_send = cloud.infer_second_model(first_feature_vector_for_send, first_split_layer_index, second_split_layer_index)
            second_model_inference_time = time.perf_counter()
            
            ## Third model : second_split_layer_index から 最後の層 (self.llm_config.num_decoder_layers) まで推論する
            output = edge.infer_third_model(second_feature_vector_for_send, second_split_layer_index)
            third_model_inference_time = time.perf_counter()

            ## 推論結果のロジットから次のトークンを選択する
            next_tokens = edge.sample_next_token(output.logits, simplified_generation_config)
            token_sampling_time = time.perf_counter()
            
            # 次のトークンを追加する
            output_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Loggerを更新する
            split_computing_logger.update(
                first_split_layer_index,
                second_split_layer_index,
                first_feature_vector_for_send,
                second_feature_vector_for_send,
                output.logits,
                inference_start_time,
                first_model_inference_time,
                second_model_inference_time,
                third_model_inference_time,
                token_sampling_time
            )

            # デトークナイズされたテキストを出力
            output_text = edge.tokenizer.decode(output_ids[0])
            yield_str = prompter.get_response(output_text) + '\n\n' + split_computing_logger.get_yield_str()
            yield yield_str

            # EOS トークンが生成されたら終了する, それ以外の場合はinput_idsを更新する
            if next_tokens[0, -1] == edge.tokenizer.eos_token_id:
                break
            else:
                input_ids = output_ids

        print(yield_str)

        # ログを出力
        split_computing_logger.save_result_to_file(output_ids, output_text)
        split_computing_logger.export_split_model_torchinfo_summary(edge, cloud)

        edge.free_memory()
        # cloud.free_memory()


    if show_ui:
        with gr.Blocks() as demo:
            if llm_config.lora_weights is None:
                gr.Markdown(f"<h1><center>Demo : Triadic Split Computing for {llm_config.base_model}</center></h1>")
            else:
                gr.Markdown(f"<h1><center>Demo : Triadic Split Computing for {llm_config.base_model} with {llm_config.lora_weights}</center></h1>")

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
                        fn=infer_each_request,
                        additional_inputs=[max_new_tokens, do_sample, temperature, top_k, top_p]
                    )

        demo.queue().launch(ssl_verify=False, server_name='0.0.0.0')

    else:
        # テキスト生成の Config
        simplified_generation_config = SimplifiedGenerationConfig(
            max_new_tokens=500,
            do_sample=False,
            use_split_past_cache=False,
            temperature=1,
            top_k=50,
            top_p=0.9
        )

        message = 'What is the difference between AI, ML and DL?' # input('Input text : ')
        for response in infer_each_request(message, None, **asdict(simplified_generation_config)):
            print(response)



if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    n = 12
    first_split_layer_indices = np.arange(n)
    second_split_layer_indices = -first_split_layer_indices
    random_seed = 42

    # LLM の Config
    ## LLaMa 2 : https://huggingface.co/meta-llama
    base_model_list_llama2 = [ 
        'meta-llama/Llama-2-7b-chat-hf',
        'meta-llama/Llama-2-13b-chat-hf',
        'meta-llama/Llama-2-70b-chat-hf',
    ]
    ## LLaMa : https://huggingface.co/huggyllama
    base_model_list_llama = [
        'huggyllama/llama-7b',
        'huggyllama/llama-13b',
        'huggyllama/llama-30b',
        'huggyllama/llama-65b'
    ]
    lora_weights_list_llama = [
        'tloen/alpaca-lora-7b',
        'Angainor/alpaca-lora-13b',
        'baseten/alpaca-30b',
        'chansung/alpaca-lora-65b'
    ]
    base_model_list_vicuna = [
        'lmsys/vicuna-7b-v1.5',
        'lmsys/vicuna-13b-v1.5'
    ]

    llm_config = LlmConfig(
        base_model=base_model_list_llama2[0],
        lora_weights=None
    )
    # llm_config = LlmConfig(
    #     base_model=base_model_list_llama[1],
    #     lora_weights=lora_weights_list_llama[1]
    # )

    second_split_layer_indices += llm_config.num_decoder_layers

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

    main(edge_split_computing_config, cloud_split_computing_config, llm_config, random_seed, False)