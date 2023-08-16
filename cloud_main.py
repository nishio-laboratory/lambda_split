import io
import threading

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
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig, SplitComputingLogger, Prompter


def cloud_main(
        cloud_split_computing_config: SplitComputingConfig,
        llm_config: LlmConfig,
        random_seed: int
    ):

    # Edge と Cloud のインスタンス
    cloud = Cloud(cloud_split_computing_config, llm_config)

    # 乱数生成器
    rng = np.random.default_rng(random_seed)

    # FastAPI のインスタンス
    app = FastAPI()

    # 排他制御のためのロックを作成
    lock = threading.Lock()

    @app.post("/infer_cloud_init/")
    def infer_init(request: Request):
        # 毎推論時に呼び出す必要がある初期化処理
        cloud.free_memory()
        cloud.init_inference()

    @app.post("/infer_cloud/")
    async def infer_each_request(request: Request):
        with lock:
            # Triadic split computing : edge -> cloud -> edge
            ## 分割するレイヤの箇所を乱数で決める
            first_split_layer_index = rng.choice(cloud.first_split_layer_indices)
            second_split_layer_index = rng.choice(cloud.second_split_layer_indices)

            ## First model を受け取る
            byte_data = await request.body()
            first_feature_vector_for_send_numpy = np.frombuffer(byte_data, dtype=np.float16).copy().reshape(1, -1, llm_config.num_embed_dims)
            first_feature_vector_for_send = torch.from_numpy(first_feature_vector_for_send_numpy).to(cloud.device)

            ## Second model : first_split_layer_index から second_split_layer_index の層まで推論する
            second_feature_vector_for_send = cloud.infer_second_model(first_feature_vector_for_send, first_split_layer_index, second_split_layer_index)
            second_feature_vector_for_send = second_feature_vector_for_send.cpu().numpy().astype(np.float16)
            
            return StreamingResponse(io.BytesIO(second_feature_vector_for_send.tobytes()), media_type="application/octet-stream")
        

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    n = 4
    first_split_layer_indices = np.array([n])
    second_split_layer_indices = np.array([-n])
    random_seed = 42

    # Cloud での SplitComputingConfig
    cloud_split_computing_config = SplitComputingConfig(
        device='mps',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=False,
    )

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

    llm_config = LlmConfig(
        base_model=base_model_list_llama2[0],
        lora_weights=None
    )
    # llm_config = LlmConfig(
    #     base_model=base_model_list_llama[1],
    #     lora_weights=lora_weights_list_llama[1]
    # )

    second_split_layer_indices += llm_config.num_decoder_layers

    cloud_main(cloud_split_computing_config, llm_config, random_seed)