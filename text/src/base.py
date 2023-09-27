import io
import random
from typing import List, Union
import gc

import numpy as np
import torch
from peft import PeftModel

from src.split_models import FirstLlamaForCausalLM, SecondLlamaForCausalLM, ThirdLlamaForCausalLM
from src.utils import SplitComputingConfig, LlmConfig


class Base:
    def __init__(
            self,
            split_computing_config: SplitComputingConfig,
            llm_config: LlmConfig
    ) -> None:
        self.split_computing_config = split_computing_config

        self.first_split_layer_indices = sorted(list(set(split_computing_config.first_split_layer_indices)))
        self.second_split_layer_indices = sorted(list(set(split_computing_config.second_split_layer_indices)))

        self.min_first_split_layer_index = min(self.first_split_layer_indices)
        self.max_first_split_layer_index = max(self.first_split_layer_indices)

        self.min_second_split_layer_index = min(self.second_split_layer_indices)
        self.max_second_split_layer_index = max(self.second_split_layer_indices)

        self.num_first_split_layer_indices = len(self.first_split_layer_indices)
        self.num_second_split_layer_indices = len(self.second_split_layer_indices)

        self.llm_config = llm_config
        
        assert 0 <= self.min_first_split_layer_index
        assert self.max_first_split_layer_index <= self.llm_config.num_decoder_layers

        assert 0 <= self.min_second_split_layer_index
        assert self.max_second_split_layer_index <= self.llm_config.num_decoder_layers

        if split_computing_config.is_max_first_less_than_min_second:
            assert self.max_first_split_layer_index <= self.min_second_split_layer_index

        self.do_replace_unused_layers_with_identity = split_computing_config.do_replace_unused_layers_with_identity

        self.device = split_computing_config.device
        self.dtype = torch.float if split_computing_config.device == 'cpu' else torch.half

        # 乱数生成器
        self.random_seed = self.split_computing_config.random_seed
        self.rng = np.random.default_rng(self.split_computing_config.random_seed)

        if "cuda" in self.device:
            assert torch.cuda.is_available()
        elif self.device == "mps":
            assert torch.backends.mps.is_available()


    def load_model(
            self, 
            position: str
        ) -> Union[FirstLlamaForCausalLM, SecondLlamaForCausalLM, ThirdLlamaForCausalLM, PeftModel]:
        assert position in ['first', 'second', 'third']

        if position == 'first':
            position_model = FirstLlamaForCausalLM
        elif position == 'second':
            position_model = SecondLlamaForCausalLM
        elif position == 'third':
            position_model = ThirdLlamaForCausalLM

        if self.device == "cuda":
            model = position_model.from_pretrained(
                self.llm_config.base_model,
                torch_dtype=self.dtype,
                device_map={"": self.device},
            )
            if self.llm_config.lora_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    self.llm_config.lora_weights,
                    torch_dtype=self.dtype,
                    device_map={"": self.device}
                )
        elif self.device == "mps":
            model = position_model.from_pretrained(
                self.llm_config.base_model,
                device_map={"": self.device},
                torch_dtype=self.dtype,
            )
            if self.llm_config.lora_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    self.llm_config.lora_weights,
                    device_map={"": self.device},
                    torch_dtype=self.dtype,
                )
        else:
            model = position_model.from_pretrained(
                self.llm_config.base_model, 
                device_map={"": self.device}, 
                low_cpu_mem_usage=True
            )
            if self.llm_config.lora_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    self.llm_config.lora_weights,
                    device_map={"": self.device},
                )

        model.tie_weights()

        # unwind broken decapoda-research config
        model.config.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

        return model
    
    def free_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    def fix_seed(self):
        # Python random
        random.seed(self.random_seed)
        # Numpy
        np.random.seed(self.random_seed)
        # Pytorch
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True