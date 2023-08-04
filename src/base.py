import copy
from typing import List, Union

import numpy as np
import torch
from peft import PeftModel

from src.models import FirstLlamaForCausalLM, SecondLlamaForCausalLM, ThirdLlamaForCausalLM
from src.util import SplitComputingConfig


class Base:
    def __init__(
            self,
            split_computing_config: SplitComputingConfig,
            base_model: str
    ) -> None:
        self.first_split_layer_indices = sorted(list(set(split_computing_config.first_split_layer_indices)))
        self.second_split_layer_indices = sorted(list(set(split_computing_config.second_split_layer_indices)))

        self.min_first_split_layer_index = min(self.first_split_layer_indices)
        self.max_first_split_layer_index = max(self.first_split_layer_indices)

        self.min_second_split_layer_index = min(self.second_split_layer_indices)
        self.max_second_split_layer_index = max(self.second_split_layer_indices)

        self.num_first_split_layer_indices = len(self.first_split_layer_indices)
        self.num_second_split_layer_indices = len(self.second_split_layer_indices)

        self.base_model = base_model

        if '7b' in self.base_model:
            self.num_decoder_layers = 32
            self.num_embed_dims = 4096

        elif '13b' in self.base_model:
            self.num_decoder_layers = 40
            self.num_embed_dims = 5120

        elif '30b' in self.base_model:
            self.num_decoder_layers = 60
            self.num_embed_dims = 6556

        
        assert 0 <= self.min_first_split_layer_index
        assert self.max_first_split_layer_index <= self.num_decoder_layers

        assert 0 <= self.min_second_split_layer_index
        assert self.max_second_split_layer_index <= self.num_decoder_layers

        if split_computing_config.is_max_first_less_than_min_second:
            assert self.max_first_split_layer_index <= self.min_second_split_layer_index

        self.do_replace_unused_layers_with_identity = split_computing_config.do_replace_unused_layers_with_identity

        self.device = split_computing_config.device

        if self.device == "cuda":
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

        model = position_model.from_pretrained(
            self.base_model,
            device_map={"": self.device},
            torch_dtype=torch.float16 if self.device == 'cuda' or 'mps' else torch.float32,
            low_cpu_mem_usage=True if self.device == 'cpu' else None
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
    