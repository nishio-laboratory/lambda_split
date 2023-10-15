import io
from typing import Union, List
from dataclasses import dataclass
import os
import time
import datetime

import torch
import numpy as np
import torchinfo


@dataclass
class SplitComputingConfig(object):
    device: str
    first_split_layer_indices: list
    second_split_layer_indices: list
    random_seed: int = 42
    use_split_sent_cache: bool = True
    use_past_key_values: bool = False
    do_shuffle: bool = False
    past_dropout_rate: float = 0.0
    is_max_first_less_than_min_second: bool = True
    do_replace_unused_layers_with_identity: bool = True
    wait_latency: bool = False
    bandwidth: int = None
    dropout_rate: float = 1.0
    quantize_method: str = None
    measure_tensor_size_method: bool = False # 'numpy_save'
    save_hidden_states_to_file: bool = False
    export_split_model_torchinfo_summary: bool = True

    def __post_init__(self):
        if self.wait_latency:
            assert self.measure_tensor_size_method is not None
            assert self.bandwidth is not None

        if self.use_past_key_values and (len(self.first_split_layer_indices) > 1 or len(self.second_split_layer_indices) > 1):
            raise NotImplementedError


@dataclass
class LlmConfig(object):
    base_model: str
    lora_weights: str

    def __post_init__(self):
        if '7b' in self.base_model:
            self.num_decoder_layers = 32
            self.num_embed_dims = 4096

        elif '13b' in self.base_model:
            self.num_decoder_layers = 40
            self.num_embed_dims = 5120

        elif '30b' in self.base_model:
            self.num_decoder_layers = 60
            self.num_embed_dims = 6556


@dataclass
class SimplifiedGenerationConfig(object):
    '''
    The original program was provided on the following page.
    https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/generation/configuration_utils.py#L38
    '''
    # Parameters that control the length of the output
    max_new_tokens: int = None

    # Parameters that control the generation strategy used
    do_sample: bool = True

    # Parameters for manipulation of the model output logits
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9


class SplitComputingLoggerForLlm(object):
    def __init__(
            self,
            edge_split_computing_config: SplitComputingConfig,
            cloud_split_computing_config: SplitComputingConfig,
            llm_config: LlmConfig,
            simplified_generation_config: SimplifiedGenerationConfig
        ) -> None:
        self.edge_split_computing_config = edge_split_computing_config
        self.cloud_split_computing_config = cloud_split_computing_config
        self.llm_config = llm_config
        self.simplified_generation_config = simplified_generation_config

        self.num_generated_tokens: int = 0

        self.first_split_layer_index_history: List[int] = []
        self.second_split_layer_index_history: List[int] = []

        self.first_model_inference_time_history: List[float] = []
        self.second_model_inference_time_history: List[float] = []
        self.third_model_inference_time_history: List[float] = []
        self.token_sampling_time_history: List[float] = []
        self.total_inference_time_history: List[float] = []

        self.first_feature_vector_for_send_size_history: List[int] = []
        self.second_feature_vector_for_send_size_history: List[int] = []

        self.save_datetime_str = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

        self.rng = np.random.default_rng(self.edge_split_computing_config.random_seed)

        self.start_time = time.perf_counter()

    def update(
            self,
            first_feature_vector_for_send: torch.Tensor,
            second_feature_vector_for_send: torch.Tensor,
            logits: torch.Tensor,
            inference_start_time: float,
            first_model_inference_time: float,
            second_model_inference_time: float,
            third_model_inference_time: float,
            token_sampling_time: float
        ):
        first_split_layer_index = self.rng.choice(self.edge_split_computing_config.first_split_layer_indices)
        second_split_layer_index = self.rng.choice(self.edge_split_computing_config.second_split_layer_indices)

        self.first_split_layer_index_history.append(first_split_layer_index)
        self.second_split_layer_index_history.append(second_split_layer_index)

        self.first_model_inference_time_history.append(first_model_inference_time - inference_start_time)
        self.second_model_inference_time_history.append(second_model_inference_time - first_model_inference_time)
        self.third_model_inference_time_history.append(third_model_inference_time - second_model_inference_time)
        self.token_sampling_time_history.append(token_sampling_time - third_model_inference_time)

        if self.num_generated_tokens == 0:
            self.total_inference_time_history.append(token_sampling_time - self.start_time)
        else:
            self.total_inference_time_history.append(token_sampling_time - self.previous_token_sampling_time)
        self.previous_token_sampling_time = token_sampling_time

        if self.edge_split_computing_config.measure_tensor_size_method:
            first_feature_vector_for_send_size = self.measure_tensor_size_and_save_to_file(
                first_feature_vector_for_send,
                save_dir='edge_to_cloud'
            )
            self.first_feature_vector_for_send_size_history.append(first_feature_vector_for_send_size)

            second_feature_vector_for_send_size = self.measure_tensor_size_and_save_to_file(
                second_feature_vector_for_send,
                save_dir='cloud_to_edge'
            )
            self.second_feature_vector_for_send_size_history.append(second_feature_vector_for_send_size)

            self.measure_tensor_size_and_save_to_file(
                logits[:, -1, :], 
                save_dir='logit'
            )

        self.num_generated_tokens += 1

    def get_yield_str(self) -> str:
        cur = time.perf_counter()

        return '(Split Computing Info)\n' + \
                f'Head sub-model layer indices (on local device) : {list(range(0, self.first_split_layer_index_history[-1]))}\n' + \
                f'Body sub-model layer indices (on cloud server) : {list(range(self.first_split_layer_index_history[-1], self.second_split_layer_index_history[-1]))}\n' + \
                f'Tail sub-model layer indices (on local device) : {list(range(self.second_split_layer_index_history[-1], self.llm_config.num_decoder_layers))}\n' + \
                f'({self.num_generated_tokens} tokens, {cur - self.start_time:.2f} seconds, {self.num_generated_tokens / (cur - self.start_time):.2f} tps)'

    def save_result_to_file(
            self,
            input_text: str,
            output_ids: torch.Tensor,
            output_text: str,
        ) -> None:
        end_time = time.perf_counter()
        total_time = end_time - self.start_time

        save_dir = os.path.join("log", self.save_datetime_str)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, 'output_ids.npy'), output_ids.cpu().detach().numpy())
        np.save(os.path.join(save_dir, 'first_split_layer_index_history.npy'), np.array(self.first_split_layer_index_history))
        np.save(os.path.join(save_dir, 'second_split_layer_index_history.npy'), np.array(self.second_split_layer_index_history))
        np.save(os.path.join(save_dir, 'first_model_inference_time_history.npy'), np.array(self.first_model_inference_time_history))
        np.save(os.path.join(save_dir, 'second_model_inference_time_history.npy'), np.array(self.second_model_inference_time_history))
        np.save(os.path.join(save_dir, 'third_model_inference_time_history.npy'), np.array(self.third_model_inference_time_history))
        np.save(os.path.join(save_dir, 'token_sampling_time_history.npy'), np.array(self.token_sampling_time_history))
        np.save(os.path.join(save_dir, 'first_feature_vector_for_send_size_history.npy'), np.array(self.first_feature_vector_for_send_size_history))
        np.save(os.path.join(save_dir, 'second_feature_vector_for_send_size_history.npy'), np.array(self.second_feature_vector_for_send_size_history))

        total_send_size = sum(self.first_feature_vector_for_send_size_history) / (1024 ** 2)
        total_receive_size = sum(self.second_feature_vector_for_send_size_history) / (1024 ** 2)

        save_filename = os.path.join("log", self.save_datetime_str, 'main.txt')

        with open(save_filename, 'w') as f:
            print('Edge  : ', self.edge_split_computing_config, file=f)
            print('Cloud : ', self.cloud_split_computing_config, file=f)
            print(self.llm_config, file=f)
            print(self.simplified_generation_config, file=f)
            print(file=f)
            print('Number of input tokens :', output_ids.shape[1] - self.num_generated_tokens, file=f)
            print('Number of generated tokens :', self.num_generated_tokens, file=f)
            print('Total inference time :', total_time, '(s)', file=f)
            print('Tokens per second :', self.num_generated_tokens / total_time, '(tps)', file=f)
            print(file=f)
            print('Total send size :', total_send_size, '(Mbit)', file=f)
            print('Total receive size :', total_receive_size, '(Mbit)', file=f)
            print('Average bandwidth :', (total_send_size + total_receive_size) / total_time, '(Mbit/s)', file=f)
            print(file=f)
            print('First split layer index history :', self.first_split_layer_index_history, file=f)
            print('Second split layer index history :', self.second_split_layer_index_history, file=f)
            print(file=f)
            print('First feature vector for send size history :', self.first_feature_vector_for_send_size_history, file=f)
            print('Second feature vector for send size history :', self.second_feature_vector_for_send_size_history, file=f)
            print(file=f)
            print('First model inference time history :', self.first_model_inference_time_history, file=f)
            print('Second model inference time history :', self.second_model_inference_time_history, file=f)
            print('Third model inference time history :', self.third_model_inference_time_history, file=f)
            print('Token sampling time history :', self.token_sampling_time_history, file=f)
            print('Total inference time history :', self.total_inference_time_history, file=f)
            print(file=f)
            print('Input text :', file=f)
            print(input_text, file=f)
            print(file=f)
            print('Output text :', file=f)
            print(output_text, file=f)

    def wait_communication_latency(self):
        wait_time = 0
        wait_time += self.first_feature_vector_for_send_size_history[-1] / self.edge_split_computing_config.bandwidth
        wait_time += self.second_feature_vector_for_send_size_history[-1] / self.edge_split_computing_config.bandwidth
        print(f"Communication latency : {wait_time} seconds")
        time.sleep(wait_time)

    # テンソルのシリアル化サイズ(bit)を測定する
    def measure_tensor_size_and_save_to_file(
            self,
            tensor: torch.Tensor,
            save_dir: str
        ) -> int:
        buffer = io.BytesIO()

        if self.edge_split_computing_config.save_hidden_states_to_file:
            save_filename = os.path.join("log", self.save_datetime_str, "hidden_state_files", save_dir, str(self.num_generated_tokens).zfill(3))
            
            os.makedirs(os.path.dirname(save_filename), exist_ok=True)

        if self.edge_split_computing_config.measure_tensor_size_method == 'numpy_save':
            tensor = tensor.to('cpu').detach().numpy().copy().astype(np.float16)
            np.save(buffer, tensor, allow_pickle=False)

            if self.edge_split_computing_config.save_hidden_states_to_file:
                np.save(save_filename, tensor, allow_pickle=False)

        elif self.edge_split_computing_config.measure_tensor_size_method == 'numpy_savez_compressed':
            tensor = tensor.to('cpu').detach().numpy().copy().astype(np.float16)
            np.savez_compressed(buffer, tensor)

            if self.edge_split_computing_config.save_hidden_states_to_file:
                np.savez_compressed(save_filename, tensor)
            
        elif self.edge_split_computing_config.measure_tensor_size_method == 'torch':
            torch.save(tensor, buffer)
            
            if self.edge_split_computing_config.save_hidden_states_to_file:
                torch.save(tensor, save_filename)

        byte_size = len(buffer.getvalue())
        bit_size = byte_size * 8

        del buffer
        
        return bit_size

    def export_split_model_torchinfo_summary(
            self,
            edge,
            cloud
        ) -> None:
        dummy_sequence_length = 50
        dummy_input_ids = torch.randint(0, 1000, (1, dummy_sequence_length)).long()
        dummy_inputs_embeds = torch.rand(1, dummy_sequence_length, self.llm_config.num_embed_dims)

        export_dir = os.path.join("log", self.save_datetime_str, "torchinfo_summary")
        os.makedirs(export_dir, exist_ok=True)
        
        with open(os.path.join(export_dir, f'first_model.txt'), 'w') as f:
            try:
                f.write(f'First  : {list(range(0, edge.max_first_split_layer_index))}')
                f.write('\n\n')
                f.write(repr(edge.first_model))
                f.write('\n\n')
                f.write(repr(torchinfo.summary(
                    edge.first_model, 
                    input_data=dummy_input_ids.long().to(edge.device),
                    depth=10, 
                    col_width=50, 
                    first_split_layer_index=edge.max_first_split_layer_index
                )))
            except Exception as e:
                f.write(repr(e))

        with open(os.path.join(export_dir, f'second_model.txt'), 'w') as f:
            try:
                f.write(f'Second : {list(range(cloud.min_first_split_layer_index, cloud.max_second_split_layer_index))}')
                f.write('\n\n')
                f.write(repr(cloud.second_model))
                f.write('\n\n')
                f.write(repr(torchinfo.summary(
                    cloud.second_model, 
                    input_data=dummy_inputs_embeds.half().to(cloud.device) if cloud.device == 'cuda' or 'mps' else dummy_inputs_embeds.float().to(cloud.device),
                    depth=10, 
                    col_width=50,
                    first_split_layer_index=cloud.min_first_split_layer_index,
                    second_split_layer_index=cloud.max_second_split_layer_index
                )))
            except Exception as e:
                f.write(repr(e))

        with open(os.path.join(export_dir, f'third_model.txt'), 'w') as f:
            try:
                f.write(f'Third  : {list(range(edge.min_second_split_layer_index, edge.llm_config.num_decoder_layers))}')
                f.write('\n\n')
                f.write(repr(edge.third_model))
                f.write('\n\n')
                f.write(repr(torchinfo.summary(
                    edge.third_model, 
                    input_data=dummy_inputs_embeds.half().to(edge.device) if edge.device == 'cuda' or 'mps' else dummy_inputs_embeds.float().to(edge.device),
                    depth=10, 
                    col_width=50,
                    second_split_layer_index=edge.min_second_split_layer_index
                )))
            except Exception as e:
                f.write(repr(e))