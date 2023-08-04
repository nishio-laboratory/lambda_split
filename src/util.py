import io
from typing import Union
from dataclasses import dataclass
import os

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
    do_shuffle: bool = True
    measure_tensor_size_method: bool = 'numpy_save'
    is_max_first_less_than_min_second: bool = True
    do_replace_unused_layers_with_identity: bool = True
    wait_latency: bool = False
    bandwidth: int = None
    dropout_rate: float = 1.0
    quantize_method: str = None
    save_split_model_output_to_file: bool = False

    def __post_init__(self):
        if self.wait_latency:
            assert self.measure_tensor_size_method is not None
            assert self.bandwidth is not None
        

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
    use_split_past_cache: bool = False

    # Parameters for manipulation of the model output logits
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self):
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


def export_split_model_torchinfo_summary(base_model, edge, cloud, export_dir: str = 'torchinfo_summary_log') -> None:
    dummy_sequence_length = 50
    dummy_input_ids = torch.randint(0, 1000, (1, dummy_sequence_length))
    dummy_inputs_embeds = torch.rand((1, dummy_sequence_length, edge.num_embed_dims))

    export_dir = os.path.join(export_dir, f'{base_model}_{edge.first_split_layer_indices}_{edge.second_split_layer_indices}')
    os.makedirs(export_dir, exist_ok=True)

    with open(os.path.join(export_dir, f'first.txt'), 'w') as f:
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

    with open(os.path.join(export_dir, f'second.txt'), 'w') as f:
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

    with open(os.path.join(export_dir, f'third.txt'), 'w') as f:
        f.write(f'Third  : {list(range(edge.min_second_split_layer_index, edge.num_decoder_layers))}')
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