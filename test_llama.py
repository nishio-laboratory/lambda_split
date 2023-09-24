import torch
import torchinfo
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.generation import LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel
from tqdm import tqdm

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from src.utils import Prompter

import time

'''
モデル可視化 : https://take-tech-engineer.com/pytorch-model-display/
中間層出力取得 : https://qiita.com/kzkadc/items/bb1bd536da5a5f8f488a
'''

class MyLlamaForCausalLM(LlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


device = "cuda"

base_model = 'huggyllama/llama-13b'
lora_weights = "Angainor/alpaca-lora-13b"

tokenizer = LlamaTokenizer.from_pretrained(base_model)
load_8bit = False

prompt_template = ""
prompter = Prompter(prompt_template)

if device == "cuda":
    model = MyLlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto" 
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
elif device == "mps":
    model = MyLlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = MyLlamaForCausalLM.from_pretrained(
        base_model, 
        device_map={"": device}, 
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        device_map={"": device},
    )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()
# model = torch.compile(model)
print(model)

# import copy
# model = copy.deepcopy(model)


import copy

model.base_model.model.model.layers = torch.nn.Sequential(
    *model.base_model.model.model.layers,
)



with torch.no_grad():
    start = time.time()
    instruction = "Tell me about Japan."
    input = None
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig()

    print(input_ids, type(input_ids))
    end = time.time()
    print(f'Tokenize time : {end - start}')

    max_new_tokens = 200

    for i in tqdm(range(max_new_tokens)):
        output = model(input_ids, use_cache=False)

        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, descending=True, dim=-1)
        
        if sorted_ids[0] == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)

    print(tokenizer.decode(input_ids[0]))

    while True:
        time.sleep(1)

    start = time.time()

    logits_processor = model._get_logits_processor(
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        input_ids_seq_length=None,
        encoder_input_ids=None,
        bad_words_ids=[[tokenizer.unk_token_id]],
        min_length=None,
        max_length=20,
        eos_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=tokenizer.bos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        prefix_allowed_tokens_fn=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        remove_invalid_values=None,
        exponential_decay_length_penalty=None,
        logits_processor=LogitsProcessorList(),
        renormalize_logits=None,
    )

    s = model.greedy_search(
        input_ids=input_ids,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
    )

    output = tokenizer.decode(s)
    print(prompter.get_response(output))

    end = time.time()
    print(end - start)

    num_input_tokens = input_ids.shape[1]
    num_total_tokens = len(s)
    num_generated_tokens = num_total_tokens - num_input_tokens
    
    print('########## Not filtered ##########')
    print(f'{num_input_tokens=}, {num_generated_tokens=}, {num_total_tokens=}')
    
    # 特殊トークンをフィルタリングします
    print('########## Filtered ##########')
    filtered_s = [token for token in s if token not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]]
    num_total_tokens = len(filtered_s)
    num_generated_tokens = num_total_tokens - num_input_tokens
    print(f'filtered : {num_input_tokens=}, {num_generated_tokens=}, {num_total_tokens=}')