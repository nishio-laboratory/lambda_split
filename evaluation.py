import io
import os
import time

import numpy as np
import torch
from tqdm import tqdm
import gradio as gr
from dataclasses import asdict

from src.cloud import Cloud
from src.edge import Edge
from src.utils import SplitComputingConfig, LlmConfig, SimplifiedGenerationConfig, SplitComputingLogger, Prompter


# fine-tuning したモデルの中間層出力を fine-tuning していないモデルで推論した場合の評価
def infer_finetuned_model_from_non_finetuned_feature_vector(first_split_layer_indices, second_split_layer_indices, random_seed, log_dir):
    # Edge での SplitComputingConfig
    edge_split_computing_config = SplitComputingConfig(
        device='cuda',
        first_split_layer_indices=first_split_layer_indices,
        second_split_layer_indices=second_split_layer_indices,
        random_seed=random_seed,
        use_split_sent_cache=False,
    )

    # Cloud での SplitComputingConfig
    cloud_split_computing_config = SplitComputingConfig(
        device='cuda',
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
        'huggyllama/llama-65b',
        # 'decapoda-research/llama-7b-hf',
        # 'decapoda-research/llama-13b-hf',
        # 'decapoda-research/llama-30b-hf',
        # 'decapoda-research/llama-65b-hf'
    ]
    lora_weights_list_llama = [
        'tloen/alpaca-lora-7b',
        'Angainor/alpaca-lora-13b',
        'baseten/alpaca-30b',
        'chansung/alpaca-lora-65b'
    ]

    # llm_config = LlmConfig(
    #     base_model=base_model_list_llama2[0],
    #     lora_weights=None
    # )
    llm_config = LlmConfig(
        base_model=base_model_list_llama[0],
        lora_weights=None
    )

    # Edge と Cloud のインスタンス
    edge = Edge(edge_split_computing_config, llm_config)
    cloud = Cloud(cloud_split_computing_config, llm_config)

    # 乱数生成器
    rng = np.random.default_rng(random_seed)


    def infer_from_first_feature_vector(message, history, max_new_tokens, do_sample, temperature, top_k, top_p, **kwargs):
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
            ### first_split_layer
            first_split_layer_relative_index = rng.integers(0, edge.num_first_split_layer_indices)
            first_split_layer_index = edge.first_split_layer_indices[first_split_layer_relative_index]
            ### second_split_layer
            second_split_layer_relative_index = rng.integers(0, edge.num_second_split_layer_indices)
            second_split_layer_index = edge.second_split_layer_indices[second_split_layer_relative_index]

            inference_start_time = time.perf_counter()

            ## First model : 0 から first_split_layer_index の層まで推論する
            first_feature_vector_for_send_filename = os.path.join(log_dir, 'hidden_state_files', 'edge_to_cloud', str(idx).zfill(3) + '.npy')
            first_feature_vector_for_send = torch.from_numpy(np.load(first_feature_vector_for_send_filename))
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
            try:
                output_ids = torch.cat([input_ids, next_tokens], dim=-1)
            except UnboundLocalError:
                output_ids = next_tokens

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
            yield_str = output_text
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




    def infer_from_second_feature_vector(message, history, max_new_tokens, do_sample, temperature, top_k, top_p, **kwargs):
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
            ### first_split_layer
            first_split_layer_relative_index = rng.integers(0, edge.num_first_split_layer_indices)
            first_split_layer_index = edge.first_split_layer_indices[first_split_layer_relative_index]
            ### second_split_layer
            second_split_layer_relative_index = rng.integers(0, edge.num_second_split_layer_indices)
            second_split_layer_index = edge.second_split_layer_indices[second_split_layer_relative_index]

            inference_start_time = time.perf_counter()

            ## First model : 0 から first_split_layer_index の層まで推論する
            first_feature_vector_for_send_filename = os.path.join(log_dir, 'hidden_state_files', 'edge_to_cloud', str(idx).zfill(3) + '.npy')
            first_feature_vector_for_send = torch.from_numpy(np.load(first_feature_vector_for_send_filename))
            first_model_inference_time = time.perf_counter()

            ## Second model : first_split_layer_index から second_split_layer_index の層まで推論する
            second_feature_vector_for_send_filename = os.path.join(log_dir, 'hidden_state_files', 'cloud_to_edge', str(idx).zfill(3) + '.npy')
            second_feature_vector_for_send = torch.from_numpy(np.load(second_feature_vector_for_send_filename))
            second_model_inference_time = time.perf_counter()
            
            ## Third model : second_split_layer_index から 最後の層 (self.llm_config.num_decoder_layers) まで推論する
            output = edge.infer_third_model(second_feature_vector_for_send, second_split_layer_index)
            third_model_inference_time = time.perf_counter()

            ## 推論結果のロジットから次のトークンを選択する
            next_tokens = edge.sample_next_token(output.logits, simplified_generation_config)
            token_sampling_time = time.perf_counter()
            
            # 次のトークンを追加する
            try:
                output_ids = torch.cat([input_ids, next_tokens], dim=-1)
            except UnboundLocalError:
                output_ids = next_tokens

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
            yield_str = output_text
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


    # テキスト生成の Config
    simplified_generation_config = SimplifiedGenerationConfig(
        max_new_tokens=76,
        do_sample=False,
        use_split_past_cache=False,
        temperature=1,
        top_k=50,
        top_p=0.9
    )

    message = 'Please explain the difference among artificial intelligence, machine learning, and deep learning.' # input('Input text : ')
    for response in infer_from_first_feature_vector(message, None, **asdict(simplified_generation_config)):
        print(response)

    for response in infer_from_second_feature_vector(message, None, **asdict(simplified_generation_config)):
        print(response)

    del edge, cloud
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # first, second = {0}, {0} or {32}, {32} or {0}, {32} の場合、decoder layersは分割されない
    # first == second の場合、2分割になる
    # first != second の場合、3分割になる
    for n in [8]:
        first_split_layer_indices = np.array([n])
        second_split_layer_indices = np.array([-n]) + 32

        log_dir = os.path.join('log', f'7b_{n}_ft')

        infer_finetuned_model_from_non_finetuned_feature_vector(first_split_layer_indices, second_split_layer_indices, 42, log_dir)



## 定性的評価


## 定量的評価




# 通信量の評価



# 推論時間の評価