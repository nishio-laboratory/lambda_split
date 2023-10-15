import time
import requests
import argparse

import numpy as np
import torch
from tqdm import tqdm
from dataclasses import asdict

from src.edge import Edge
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig, SplitComputingLoggerForLlm


def edge_main(
        edge_split_computing_config: SplitComputingConfig,
        cloud_split_computing_config: SplitComputingConfig,
        llm_config: LlmConfig,
        show_ui: bool,
        url: str
    ):

    # Edge と Cloud のインスタンス
    edge = Edge(edge_split_computing_config, llm_config)


    def infer_each_request(input_text, history, max_new_tokens, do_sample, temperature, top_k, top_p, **kwargs):
        # 毎推論時に呼び出す必要がある初期化処理
        edge.init_inference()

        # テキスト生成の Config
        simplified_generation_config = SimplifiedGenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        input_ids = edge.tokenizer(input_text, return_tensors="pt")["input_ids"].to(edge.device)
        input_length = input_ids.shape[-1]

        split_computing_logger = SplitComputingLoggerForLlm(
            edge_split_computing_config, 
            cloud_split_computing_config, 
            llm_config, 
            simplified_generation_config
        )

        requests.post(f"http://{url}/infer_cloud_init/")

        for idx in tqdm(range(simplified_generation_config.max_new_tokens)):
            print(idx)

            # Triadic split computing : edge -> cloud -> edge
            inference_start_time = time.perf_counter()

            ## First model : 0 から first_split_layer_index の層まで推論する
            first_feature_vector_for_send = edge.infer_first_model(input_ids)
            first_feature_vector_for_send_numpy = first_feature_vector_for_send.cpu().numpy().astype(np.float16)
            first_model_inference_time = time.perf_counter()

            ## Second model に送信する
            response = requests.post(f"http://{url}/infer_cloud/", data=first_feature_vector_for_send_numpy.tobytes())
            second_model_inference_time = time.perf_counter()

            second_feature_vector_for_send_numpy = np.frombuffer(response.content, dtype=np.float16).copy().reshape(1, -1, llm_config.num_embed_dims)
            second_feature_vector_for_send = torch.from_numpy(second_feature_vector_for_send_numpy).to(edge.device)
            
            ## Third model : second_split_layer_index から 最後の層 (self.llm_config.num_decoder_layers) まで推論する
            output = edge.infer_third_model(second_feature_vector_for_send)
            third_model_inference_time = time.perf_counter()

            ## 推論結果のロジットから次のトークンを選択する
            next_tokens = edge.sample_next_token(output.logits, simplified_generation_config)
            token_sampling_time = time.perf_counter()
            
            # 次のトークンを追加する
            output_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Loggerを更新する
            split_computing_logger.update(
                first_feature_vector_for_send,
                second_feature_vector_for_send,
                output.logits,
                inference_start_time,
                first_model_inference_time,
                second_model_inference_time,
                third_model_inference_time,
                token_sampling_time
            )

            # EOS トークンが生成されたら終了する, それ以外の場合はinput_idsを更新する
            if next_tokens[0, -1] == edge.tokenizer.eos_token_id:
                break
            
            input_ids = output_ids

            # デトークナイズされたテキストを出力
            output_text = edge.tokenizer.decode(output_ids[0, input_length:]).strip()
            yield_str = output_text + '\n\n' + split_computing_logger.get_yield_str()
            
            yield yield_str

        print(yield_str)

        # ログを出力
        split_computing_logger.save_result_to_file(input_text, output_ids, output_text)

        requests.post(f"http://{url}/infer_cloud_end/")


    if show_ui:
        import gradio as gr
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
            max_new_tokens=300,
            do_sample=False,
            temperature=1,
            top_k=50,
            top_p=0.9
        )

        input_text = 'What is the difference between AI, ML and DL?'
        for response in infer_each_request(input_text, None, **asdict(simplified_generation_config)):
            print(response)



if __name__ == '__main__':
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--edge_device', type=str, default=default_device, help='cuda or mps or cpu')
    parser.add_argument('--cloud_device', type=str, default=default_device, help='cuda or mps or cpu')
    parser.add_argument('--first_split_layer', nargs='+', type=int, default=[1], help='--first_split_layer 1, or --first_split_layer 1 2 3')
    parser.add_argument('--second_split_layer', nargs='+', type=int, default=[-1], help='--second_split_layer 31, or --second_split_layer -1')
    parser.add_argument('-n', type=int, help='Top and bottom n layers inferred at the edge: -n 1')
    parser.add_argument('--disable_past_caching', action='store_true', help='Disable past caching')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='base model name')
    parser.add_argument('--lora_weights', type=str, default=None, help='lora weights name')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument('--no_gui', action='store_true', help='Disable Gradio GUI')
    parser.add_argument('--url', type=str, default='192.168.1.221:7860', help='URL')
    args = parser.parse_args()
    print(args)

    edge_device = args.edge_device
    cloud_device = args.cloud_device

    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    if args.n is not None:
        n = args.n
        first_split_layer_indices = np.array([n])
        second_split_layer_indices = np.array([-n])
    else:
        first_split_layer_indices = np.array(args.first_split_layer)
        second_split_layer_indices = np.array(args.second_split_layer)

    # first_split_layer_indices に 負の数が存在している場合はエラー
    if np.any(first_split_layer_indices < 0):
        raise ValueError("There is a negative number in first_split_layer_indices.")
    
    # second_split_layer_indices に 正の数と負の数が混在している場合はエラー
    if np.any(second_split_layer_indices > 0) and np.any(second_split_layer_indices < 0):
        raise ValueError("There is a mixture of positive and negative numbers in second_split_layer_indices.")
    
    past_caching = not args.disable_past_caching
    random_seed = args.random_seed
    show_ui = not args.no_gui
    url = args.url

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
    base_model_list_elyza = [
        'elyza/ELYZA-japanese-Llama-2-7b-instruct',
        'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct'
    ]

    llm_config = LlmConfig(
        base_model=args.base_model,
        lora_weights=args.lora_weights
    )
    # llm_config = LlmConfig(
    #     base_model=base_model_list_llama[1],
    #     lora_weights=lora_weights_list_llama[1]
    # )

    # second_split_layer_indices が 全て負の数の場合は、+= llm_config.num_decoder_layers
    if np.all(second_split_layer_indices <= 0):
        second_split_layer_indices += llm_config.num_decoder_layers

    # Edge での SplitComputingConfig
    edge_split_computing_config = SplitComputingConfig(
        device=edge_device,
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=past_caching,
        use_past_key_values=False,
    )

    # Cloud での SplitComputingConfig
    cloud_split_computing_config = SplitComputingConfig(
        device=cloud_device,
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=past_caching,
        use_past_key_values=False,
    )

    edge_main(edge_split_computing_config, cloud_split_computing_config, llm_config, show_ui, url)