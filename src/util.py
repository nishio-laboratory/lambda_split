import io
from typing import Union

import torch
import numpy as np
import torch.nn.functional as F
import torchinfo
from transformers import GenerationConfig


class SplitComputingConfig(object):
    def __init__(
            self,
            device: str,
            first_split_layer_indices: list,
            second_split_layer_indices: list,
            random_seed: int = 42,
            use_split_sent_cache: bool = True,
            measure_tensor_size_method: bool = 'numpy_save',
            is_max_first_less_than_min_second: bool = True,
            do_replace_unused_layers_with_identity: bool = True,
            wait_latency: bool = False,
            bandwidth: int = None,
            dropout_rate: float = 1.0,
            quantize_method: str = None,
            save_split_model_output_to_file: bool = False
    ) -> None:
        if wait_latency:
            assert measure_tensor_size_method is not None
            assert bandwidth is not None

        self.device = device
        self.first_split_layer_indices = first_split_layer_indices
        self.second_split_layer_indices = second_split_layer_indices
        self.random_seed = random_seed
        self.use_split_sent_cache = use_split_sent_cache
        self.measure_tensor_size_method = measure_tensor_size_method
        self.is_max_first_less_than_min_second = is_max_first_less_than_min_second
        self.do_replace_unused_layers_with_identity = do_replace_unused_layers_with_identity
        self.wait_latency = wait_latency
        self.bandwidth = bandwidth
        self.dropout_rate = dropout_rate
        self.quantize_method = quantize_method
        self.save_split_model_output_to_file = save_split_model_output_to_file


class LLMConfig(object):
    def __init__(
        self,
        base_model: str = 'huggyllama/llama-7b',
        lora_weights: str = "tloen/alpaca-lora-7b"
    ) -> None:
        self.base_model = base_model
        self.lora_weights = lora_weights
        

class SimplifiedGenerationConfig(GenerationConfig):
    '''
    The original program was provided on the following page.
    https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/generation/configuration_utils.py#L38
    '''
    def __init__(
            self,
            max_new_tokens: int = None,
            do_sample: bool = True,
            use_split_past_cache: bool = False,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 0.9
        ) -> None:
        # Parameters that control the length of the output
        self.max_new_tokens = max_new_tokens

        # Parameters that control the generation strategy used
        self.do_sample = do_sample
        self.use_split_past_cache = use_split_past_cache

        # Parameters for manipulation of the model output logits
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        if self.use_split_past_cache:
            raise NotImplementedError

    def from_generation_config(
        self,
        config: GenerationConfig
    ) -> None:
        # Parameters that control the length of the output
        self.max_new_tokens = config.max_new_tokens

        # Parameters that control the generation strategy used
        self.do_sample = config.do_sample
        self.num_beams = config.num_beams
        self.use_cache = config.use_cache

        # Parameters for manipulation of the model output logits
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p

        assert self.num_beams == 1

        if self.use_split_past_cache:
            raise NotImplementedError


class Prompter(object):
    """
    The original program was provided on the following page.
    https://github.com/tloen/alpaca-lora/blob/main/utils/prompter.py
    https://github.com/tloen/alpaca-lora/blob/main/templates/alpaca.json
    """
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"    
        }

        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    


# テンソルのシリアル化サイズ(bit)を測定する関数
def measure_tensor_size(
        tensor: torch.Tensor,
        method: str = 'numpy_save'
    ) -> int:
    buffer = io.BytesIO()

    if method == 'numpy_save':
        tensor = tensor.to('cpu').detach().numpy().copy().astype(np.float16)
        np.save(buffer, tensor, allow_pickle=False)

    elif method == 'numpy_savez_compressed':
        tensor = tensor.to('cpu').detach().numpy().copy().astype(np.float16)
        np.savez_compressed(buffer, tensor)
        
    elif method == 'torch':
        torch.save(tensor, buffer)

    byte_size = len(buffer.getvalue())
    bit_size = byte_size * 8
    return bit_size


# ロジットから次のトークンを選択する関数
def select_next_token(
        logits: torch.Tensor,
        config: SimplifiedGenerationConfig
    ) -> torch.Tensor:
    do_sample = config.do_sample
    temperature = config.temperature
    top_k = config.top_k
    top_p = config.top_p

    assert len(logits.shape) == 1

    if not do_sample:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        return next_token

    # If top_k is not provided, consider all tokens
    if top_k is None:
        top_k = logits.shape[-1]
    
    # Apply temperature if given
    logits = logits / temperature

    # Apply nucleus (top-p) sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    logits = sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

    # Apply top-k sampling
    top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token



def export_torchinfo_summary(edge, cloud, input_ids):
    with open(f'torchinfo_summary_log/first_{edge.first_split_layer_indices}_{edge.second_split_layer_indices}.txt', 'w') as f:
        f.write(repr(torchinfo.summary(edge.first_model, input_data=input_ids, depth=10, col_width=50)))
    with open(f'torchinfo_summary_log/second_{edge.first_split_layer_indices}_{edge.second_split_layer_indices}.txt', 'w') as f:
        f.write(repr(torchinfo.summary(cloud.second_model, input_data=first_feature_vector, depth=10, col_width=50)))
    with open(f'torchinfo_summary_log/third_{first_split_layer_indices}_{second_split_layer_indices}.txt', 'w') as f:
        f.write(repr(torchinfo.summary(edge.third_model, input_data=second_feature_vector, depth=10, col_width=50)))