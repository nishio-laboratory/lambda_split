import time
import argparse

import numpy as np
import torch
from tqdm import tqdm
from dataclasses import asdict
import pandas as pd
from scipy.spatial.distance import cosine

from src.cloud import Cloud
from src.edge import Edge
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig, SplitComputingLoggerForLlm


dropout_method_list = ['random'] # 'block'
dropout_rate_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
tail_layers_list = [1, 4, 8]


df = pd.DataFrame(columns=['dropout_method', 'dropout_rate', 'tail_layers', 'output_text'])


for tail_layers in tail_layers_list:
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

    n = tail_layers
    first_split_layer_indices = np.array([n])
    second_split_layer_indices = np.array([-n])

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

    # second_split_layer_indices が 全て0または負の数の場合は、+= llm_config.num_decoder_layers
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

    # Edge と Cloud のインスタンス
    edge = Edge(edge_split_computing_config, llm_config)
    cloud = Cloud(cloud_split_computing_config, llm_config)

    input_text = ''
    simplified_generation_config = SimplifiedGenerationConfig(
            max_new_tokens=500,
            do_sample=False,
            temperature=1,
            top_k=50,
            top_p=0.9
    )

    for dropout_method in dropout_method_list:
        for dropout_rate in dropout_rate_list:

            # 毎推論時に呼び出す必要がある初期化処理
            edge.init_inference()
            cloud.init_inference()

            input_ids = edge.tokenizer(input_text, return_tensors="pt")["input_ids"].to(edge.device)
            input_length = input_ids.shape[-1]

            split_computing_logger = SplitComputingLoggerForLlm(
                edge_split_computing_config, 
                cloud_split_computing_config, 
                llm_config, 
                simplified_generation_config
            )

            try:
                for idx in tqdm(range(simplified_generation_config.max_new_tokens)):
                    print(idx)

                    ## Second model : first_split_layer_index から second_split_layer_index の層まで推論する
                    second_feature_vector_for_send = np.load(f'log/eavesdrop_{tail_layers}/hidden_state_files/cloud_to_edge/{str(idx).zfill(3)}.npy')
                    # シャッフルする
                    rng = np.random.default_rng()
                    second_feature_vector_for_send = rng.permutation(second_feature_vector_for_send, axis=2)
                    second_feature_vector_for_send = torch.from_numpy(second_feature_vector_for_send).to(edge.device)
                    # second_feature_vector_for_send = torch.nn.functional.dropout(second_feature_vector_for_send.float(), p=dropout_rate, training=True).half()



                    ## Third model : second_split_layer_index から 最後の層 (self.llm_config.num_decoder_layers) まで推論する
                    output = edge.infer_third_model(second_feature_vector_for_send)

                    ## 推論結果のロジットから次のトークンを選択する
                    next_tokens = edge.sample_next_token(output.logits, simplified_generation_config)
                    
                    # 次のトークンを追加する
                    output_ids = torch.cat([input_ids, next_tokens], dim=-1)

                    # EOS トークンが生成されたら終了する, それ以外の場合はinput_idsを更新する
                    if next_tokens[0, -1] == edge.tokenizer.eos_token_id:
                        break

                    input_ids = output_ids

                    # デトークナイズされたテキストを出力
                    output_text = edge.tokenizer.decode(output_ids[0, input_length:]).strip()

            except FileNotFoundError:
                pass


            # # 最後の隠れ層の平均を取ることで文のベクトルを得る
            # embeddings = output.last_hidden_state.mean(1)

            # if dropout_rate == 0.0:
            #     original_embeddings = embeddings

            # cosine_simularity = cosine(original_embeddings, embeddings)
                
            output_text = output_text.replace('\n', ' ')

            df = df.append({
                'dropout_method': dropout_method,
                'dropout_rate': dropout_rate,
                'tail_layers': tail_layers,
                'output_text': output_text
            }, ignore_index=True)

            df.to_csv(f'eavesdrop.csv', index=False)

    del edge, cloud
    torch.cuda.empty_cache()