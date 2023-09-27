import io
import threading
import argparse

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from dataclasses import asdict
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

from src.cloud import Cloud
from src.edge import Edge
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig, SplitComputingLoggerForLlm


def cloud_main(
        cloud_split_computing_config: SplitComputingConfig,
        llm_config: LlmConfig
    ):

    # Edge と Cloud のインスタンス
    cloud = Cloud(cloud_split_computing_config, llm_config)

    # FastAPI のインスタンス
    app = FastAPI()

    # 排他制御のためのロックを作成
    lock = threading.Lock()

    @app.post("/infer_cloud_init/")
    def infer_init(request: Request):
        # 毎推論時に呼び出す必要がある初期化処理
        cloud.init_inference()

    @app.post("/infer_cloud/")
    async def infer_each_request(request: Request):
        with lock:
            # Triadic split computing : edge -> cloud -> edge

            ## First model を受け取る
            byte_data = await request.body()
            first_feature_vector_for_send_numpy = np.frombuffer(byte_data, dtype=np.float16).copy().reshape(1, -1, llm_config.num_embed_dims)
            first_feature_vector_for_send = torch.from_numpy(first_feature_vector_for_send_numpy).to(cloud.device)

            ## Second model : first_split_layer_index から second_split_layer_index の層まで推論する
            second_feature_vector_for_send = cloud.infer_second_model(first_feature_vector_for_send)
            second_feature_vector_for_send = second_feature_vector_for_send.cpu().numpy().astype(np.float16)
            
            return StreamingResponse(io.BytesIO(second_feature_vector_for_send.tobytes()), media_type="application/octet-stream")
        

    uvicorn.run(app, host="0.0.0.0", port=7860)


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

    # Cloud での SplitComputingConfig
    cloud_split_computing_config = SplitComputingConfig(
        device=cloud_device,
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=past_caching,
        use_past_key_values=False,
    )

    cloud_main(cloud_split_computing_config, llm_config)