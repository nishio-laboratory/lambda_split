import copy

import numpy as np
import torch
from peft import PeftModel

from src.models import FirstLlamaForCausalLM, SecondLlamaForCausalLM, ThirdLlamaForCausalLM


class Base:
    def __init__(
            self,
            first_split_layer_indices: set,
            second_split_layer_indices: set,
            random_seed: int = 42,
            load_8bit: bool = False,
            base_model: str = 'huggyllama/llama-7b',
            lora_weights: str = "tloen/alpaca-lora-7b",
            num_decoder_layers: int = 32,
            num_embed_dims: int = 4096,
            is_max_first_less_than_min_second: bool = True, # メモリがたくさんあるならFalseでいい
            do_replace_unused_layers_with_identity: bool = True
    ) -> None:
        self.first_split_layer_indices = sorted(list(first_split_layer_indices))
        self.second_split_layer_indices = sorted(list(second_split_layer_indices))

        self.min_first_split_layer_index = min(first_split_layer_indices)
        self.max_first_split_layer_index = max(first_split_layer_indices)

        self.min_second_split_layer_index = min(second_split_layer_indices)
        self.max_second_split_layer_index = max(second_split_layer_indices)

        self.num_first_split_layer_indices = len(self.first_split_layer_indices)
        self.num_second_split_layer_indices = len(self.second_split_layer_indices)
        
        self.num_decoder_layers = num_decoder_layers
        self.num_embed_dims = num_embed_dims
        
        assert 0 <= self.min_first_split_layer_index
        assert self.max_first_split_layer_index <= self.num_decoder_layers

        assert 0 <= self.min_second_split_layer_index
        assert self.max_second_split_layer_index <= self.num_decoder_layers

        if is_max_first_less_than_min_second:
            assert self.max_first_split_layer_index <= self.min_second_split_layer_index

        self.do_replace_unused_layers_with_identity = do_replace_unused_layers_with_identity
    
        self.load_8bit = load_8bit
        self.base_model = base_model
        self.lora_weights = lora_weights

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # 乱数生成器
        self.rng = np.random.default_rng(random_seed)


    def load_model(
            self, 
            position: str
        ) -> PeftModel:
        assert position in ['first', 'second', 'third']

        if position == 'first':
            position_model = FirstLlamaForCausalLM
        elif position == 'second':
            position_model = SecondLlamaForCausalLM
        elif position == 'third':
            position_model = ThirdLlamaForCausalLM

        if self.device == "cuda":
            model = position_model.from_pretrained(
                self.base_model,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                self.lora_weights,
                torch_dtype=torch.float16,
            )
        elif self.device == "mps":
            model = position_model.from_pretrained(
                self.base_model,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                self.lora_weights,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
        else:
            model = position_model.from_pretrained(
                self.base_model, 
                device_map={"": self.device}, 
                low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                self.lora_weights,
                device_map={"": self.device},
            )

        model.tie_weights()

        # unwind broken decapoda-research config
        model.config.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not self.load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)

        return model
    