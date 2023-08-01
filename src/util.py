

from typing import Union
from transformers import GenerationConfig


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


class SplitComputingConfig(object):
    def __init__(self, **kwargs) -> None:
        self.use_split_cache = kwargs.pop("use_split_cache", True)
        self.use_past_cache = kwargs.pop("use_split_cache", True)
        self.wait_latency = kwargs.pop("use_split_cache", False)
        self.bandwidth = kwargs.pop("use_split_cache", None)
        self.dropout_rate = kwargs.pop("use_split_cache", 1)
        self.quantize_method = kwargs.pop("use_split_cache", None)
        self.save_split_model_output_to_file = kwargs.pop("save_split_model_output_to_file", False)
        


def SimplifiedGenerationConfig(GenerationConfig):
    '''
    The original program was provided on the following page.
    https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/generation/configuration_utils.py#L38
    '''
    def __init__(self, **kwargs) -> None:
        # Parameters that control the length of the output
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.use_cache = kwargs.pop("use_cache", True)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)

        assert self.num_beams == 1

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