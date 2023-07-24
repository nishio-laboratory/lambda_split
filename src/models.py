'''
LlamaModel と LlamaForCausalLM の forward メソッドを override する

transformers-4.31.0

Reference
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
'''

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import logger


class FirstLlamaModel(LlamaModel):
    def replace_unused_layers_with_identity(
            self,
            max_first_split_layer_index: int = None
    ) -> None:
        self.num_decoder_layers = len(self.layers)

        for i in range(max_first_split_layer_index, self.num_decoder_layers):
            self.layers[i] = ExtendedIdentity()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        split_first_layer_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> torch.HalfTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        self.num_decoder_layers = len(self.layers)
        print('First  :', list(range(0, split_first_layer_index)))

        for idx in range(0, split_first_layer_index):
            decoder_layer = self.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        return hidden_states

        ''' ここをコメントアウト
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        '''


class FirstLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = FirstLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        split_first_layer_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> torch.HalfTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            split_first_layer_index=split_first_layer_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

        ''' ここをコメントアウト
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
        '''



class SecondLlamaModel(LlamaModel):
    def replace_unused_layers_with_identity(
            self,
            min_first_split_layer_index: int = None,
            max_second_split_layer_index: int = None
    ) -> None:
        self.num_decoder_layers = len(self.layers)

        for i in range(0, min_first_split_layer_index):
            self.layers[i] = ExtendedIdentity()

        for i in range(max_second_split_layer_index, self.num_decoder_layers):
            self.layers[i] = ExtendedIdentity()
        
    def forward(
        self,
        input_ids: torch.HalfTensor = None,
        split_first_layer_index: int = None,
        split_second_layer_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> torch.HalfTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        self.num_decoder_layers = len(self.layers)
        print('Second :', list(range(split_first_layer_index, split_second_layer_index)))

        for idx in range(split_first_layer_index, split_second_layer_index):
            decoder_layer = self.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        return hidden_states

        ''' ここをコメントアウト
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        '''


class SecondLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = SecondLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.HalfTensor = None,
        split_first_layer_index: int = None,
        split_second_layer_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> torch.HalfTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            split_first_layer_index=split_first_layer_index,
            split_second_layer_index=split_second_layer_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

        ''' ここをコメントアウト
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
        '''



class ThirdLlamaModel(LlamaModel):
    def replace_unused_layers_with_identity(
            self,
            min_second_split_layer_index: int = None
    ) -> None:
        self.num_decoder_layers = len(self.layers)

        for i in range(0, min_second_split_layer_index):
            self.layers[i] = ExtendedIdentity()

    def forward(
        self,
        input_ids: torch.HalfTensor = None,
        split_second_layer_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        self.num_decoder_layers = len(self.layers)
        print('Third  :', list(range(split_second_layer_index, self.num_decoder_layers)))

        for idx in range(split_second_layer_index, self.num_decoder_layers):
            decoder_layer = self.layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ThirdLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = ThirdLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.HalfTensor = None,
        split_second_layer_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> torch.Tensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            split_second_layer_index=split_second_layer_index,
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



class ExtendedIdentity(torch.nn.Identity):
    # forward の引数を無視する仕様を追加
    def forward(self, input, *args, **kwargs):
        return input


''' model の概要
PeftModelForCausalLM(
    (base_model): LoraModel(
        (model): LlamaForCausalLM(
            (model): LlamaModel(
                (embed_tokens): Embedding(32000, 4096, padding_idx=0)
                (layers): ModuleList(
                    (0-31): 32 x LlamaDecoderLayer(
                        (self_attn): LlamaAttention(
                            (q_proj): Linear(
                                in_features=4096, out_features=4096, bias=False
                                (lora_dropout): ModuleDict(
                                    (default): Dropout(p=0.05, inplace=False)
                                )
                                (lora_A): ModuleDict(
                                    (default): Linear(in_features=4096, out_features=16, bias=False)
                                )
                                (lora_B): ModuleDict(
                                    (default): Linear(in_features=16, out_features=4096, bias=False)
                                )
                                (lora_embedding_A): ParameterDict()
                                (lora_embedding_B): ParameterDict()
                            )
                            (k_proj): Linear(
                                in_features=4096, out_features=4096, bias=False
                                (lora_dropout): ModuleDict(
                                    (default): Dropout(p=0.05, inplace=False)
                                )
                                (lora_A): ModuleDict(
                                    (default): Linear(in_features=4096, out_features=16, bias=False)
                                )
                                (lora_B): ModuleDict(
                                    (default): Linear(in_features=16, out_features=4096, bias=False)
                                )
                                (lora_embedding_A): ParameterDict()
                                (lora_embedding_B): ParameterDict()
                            )
                            (v_proj): Linear(
                                in_features=4096, out_features=4096, bias=False
                                (lora_dropout): ModuleDict(
                                    (default): Dropout(p=0.05, inplace=False)
                                )
                                (lora_A): ModuleDict(
                                    (default): Linear(in_features=4096, out_features=16, bias=False)
                                )
                                (lora_B): ModuleDict(
                                    (default): Linear(in_features=16, out_features=4096, bias=False)
                                )
                                (lora_embedding_A): ParameterDict()
                                (lora_embedding_B): ParameterDict()
                            )
                            (o_proj): Linear(
                                in_features=4096, out_features=4096, bias=False
                                (lora_dropout): ModuleDict(
                                    (default): Dropout(p=0.05, inplace=False)
                                )
                                (lora_A): ModuleDict(
                                    (default): Linear(in_features=4096, out_features=16, bias=False)
                                )
                                (lora_B): ModuleDict(
                                    (default): Linear(in_features=16, out_features=4096, bias=False)
                                )
                                (lora_embedding_A): ParameterDict()
                                (lora_embedding_B): ParameterDict()
                            )
                            (rotary_emb): LlamaRotaryEmbedding()
                        )
                        (mlp): LlamaMLP(
                            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
                            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
                            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
                            (act_fn): SiLUActivation()
                        )
                        (input_layernorm): LlamaRMSNorm()
                        (post_attention_layernorm): LlamaRMSNorm()
                    )
                )
                (norm): LlamaRMSNorm()
            )
            (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
        )
    )
)
'''



''' torchinfo.summary
=====================================================================================================================================================================
Layer (type:depth-idx)                                            Output Shape                                       Param #
=====================================================================================================================================================================
PeftModelForCausalLM                                              [1, 32, 86, 128]                                   --
├─LoraModel: 1-1                                                  [1, 32, 86, 128]                                   --
│    └─LlamaForCausalLM: 2-1                                      --                                                 --
│    │    └─LlamaModel: 3-1                                       [1, 32, 86, 128]                                   --
│    │    │    └─Embedding: 4-1                                   [1, 86, 4096]                                      (131,072,000)
│    │    │    └─ModuleList: 4-2                                  --                                                 --
│    │    │    │    └─LlamaDecoderLayer: 5-1                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-1                      [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-2                    [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-1                       [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-1              --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-1            [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-2              --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-2             [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-3              --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-3             [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-2                       [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-4              --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-4            [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-5              --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-5             [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-6              --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-6             [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-3                       [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-7              --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-7            [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-8              --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-8             [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-9              --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-9             [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-4         [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-5                       [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-10             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-10           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-11             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-11            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-12             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-12            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-3                      [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-4                          [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-6                       [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-7               [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-8                       [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-9                       [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-2                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-5                      [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-6                    [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-10                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-13             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-13           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-14             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-14            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-15             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-15            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-11                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-16             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-16           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-17             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-17            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-18             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-18            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-12                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-19             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-19           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-20             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-20            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-21             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-21            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-13        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-14                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-22             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-22           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-23             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-23            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-24             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-24            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-7                      [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-8                          [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-15                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-16              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-17                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-18                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-3                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-9                      [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-10                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-19                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-25             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-25           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-26             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-26            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-27             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-27            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-20                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-28             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-28           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-29             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-29            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-30             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-30            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-21                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-31             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-31           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-32             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-32            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-33             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-33            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-22        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-23                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-34             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-34           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-35             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-35            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-36             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-36            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-11                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-12                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-24                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-25              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-26                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-27                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-4                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-13                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-14                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-28                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-37             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-37           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-38             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-38            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-39             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-39            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-29                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-40             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-40           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-41             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-41            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-42             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-42            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-30                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-43             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-43           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-44             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-44            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-45             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-45            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-31        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-32                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-46             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-46           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-47             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-47            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-48             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-48            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-15                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-16                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-33                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-34              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-35                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-36                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-5                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-17                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-18                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-37                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-49             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-49           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-50             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-50            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-51             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-51            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-38                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-52             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-52           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-53             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-53            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-54             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-54            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-39                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-55             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-55           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-56             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-56            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-57             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-57            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-40        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-41                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-58             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-58           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-59             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-59            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-60             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-60            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-19                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-20                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-42                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-43              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-44                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-45                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-6                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-21                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-22                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-46                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-61             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-61           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-62             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-62            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-63             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-63            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-47                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-64             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-64           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-65             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-65            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-66             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-66            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-48                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-67             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-67           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-68             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-68            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-69             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-69            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-49        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-50                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-70             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-70           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-71             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-71            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-72             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-72            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-23                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-24                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-51                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-52              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-53                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-54                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-7                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-25                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-26                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-55                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-73             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-73           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-74             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-74            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-75             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-75            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-56                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-76             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-76           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-77             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-77            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-78             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-78            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-57                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-79             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-79           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-80             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-80            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-81             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-81            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-58        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-59                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-82             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-82           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-83             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-83            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-84             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-84            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-27                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-28                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-60                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-61              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-62                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-63                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-8                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-29                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-30                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-64                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-85             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-85           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-86             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-86            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-87             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-87            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-65                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-88             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-88           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-89             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-89            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-90             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-90            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-66                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-91             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-91           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-92             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-92            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-93             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-93            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-67        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-68                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-94             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-94           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-95             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-95            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-96             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-96            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-31                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-32                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-69                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-70              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-71                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-72                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-9                      [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-33                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-34                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-73                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-97             --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-97           [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-98             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-98            [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-99             --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-99            [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-74                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-100            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-100          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-101            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-101           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-102            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-102           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-75                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-103            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-103          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-104            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-104           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-105            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-105           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-76        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-77                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-106            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-106          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-107            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-107           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-108            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-108           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-35                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-36                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-78                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-79              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-80                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-81                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-10                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-37                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-38                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-82                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-109            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-109          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-110            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-110           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-111            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-111           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-83                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-112            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-112          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-113            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-113           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-114            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-114           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-84                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-115            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-115          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-116            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-116           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-117            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-117           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-85        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-86                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-118            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-118          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-119            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-119           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-120            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-120           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-39                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-40                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-87                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-88              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-89                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-90                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-11                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-41                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-42                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-91                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-121            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-121          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-122            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-122           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-123            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-123           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-92                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-124            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-124          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-125            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-125           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-126            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-126           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-93                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-127            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-127          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-128            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-128           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-129            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-129           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-94        [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-95                      [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-130            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-130          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-131            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-131           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-132            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-132           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-43                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-44                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-96                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-97              [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-98                      [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-99                      [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-12                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-45                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-46                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-100                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-133            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-133          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-134            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-134           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-135            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-135           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-101                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-136            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-136          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-137            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-137           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-138            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-138           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-102                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-139            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-139          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-140            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-140           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-141            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-141           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-103       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-104                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-142            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-142          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-143            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-143           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-144            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-144           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-47                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-48                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-105                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-106             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-107                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-108                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-13                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-49                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-50                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-109                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-145            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-145          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-146            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-146           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-147            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-147           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-110                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-148            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-148          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-149            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-149           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-150            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-150           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-111                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-151            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-151          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-152            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-152           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-153            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-153           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-112       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-113                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-154            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-154          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-155            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-155           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-156            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-156           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-51                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-52                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-114                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-115             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-116                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-117                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-14                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-53                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-54                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-118                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-157            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-157          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-158            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-158           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-159            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-159           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-119                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-160            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-160          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-161            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-161           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-162            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-162           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-120                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-163            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-163          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-164            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-164           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-165            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-165           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-121       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-122                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-166            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-166          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-167            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-167           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-168            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-168           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-55                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-56                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-123                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-124             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-125                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-126                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-15                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-57                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-58                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-127                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-169            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-169          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-170            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-170           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-171            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-171           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-128                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-172            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-172          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-173            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-173           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-174            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-174           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-129                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-175            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-175          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-176            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-176           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-177            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-177           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-130       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-131                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-178            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-178          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-179            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-179           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-180            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-180           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-59                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-60                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-132                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-133             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-134                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-135                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-16                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-61                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-62                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-136                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-181            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-181          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-182            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-182           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-183            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-183           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-137                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-184            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-184          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-185            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-185           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-186            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-186           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-138                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-187            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-187          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-188            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-188           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-189            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-189           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-139       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-140                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-190            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-190          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-191            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-191           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-192            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-192           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-63                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-64                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-141                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-142             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-143                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-144                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-17                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-65                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-66                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-145                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-193            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-193          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-194            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-194           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-195            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-195           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-146                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-196            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-196          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-197            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-197           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-198            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-198           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-147                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-199            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-199          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-200            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-200           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-201            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-201           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-148       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-149                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-202            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-202          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-203            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-203           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-204            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-204           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-67                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-68                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-150                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-151             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-152                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-153                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-18                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-69                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-70                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-154                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-205            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-205          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-206            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-206           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-207            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-207           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-155                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-208            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-208          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-209            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-209           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-210            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-210           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-156                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-211            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-211          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-212            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-212           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-213            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-213           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-157       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-158                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-214            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-214          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-215            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-215           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-216            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-216           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-71                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-72                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-159                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-160             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-161                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-162                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-19                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-73                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-74                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-163                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-217            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-217          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-218            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-218           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-219            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-219           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-164                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-220            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-220          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-221            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-221           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-222            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-222           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-165                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-223            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-223          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-224            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-224           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-225            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-225           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-166       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-167                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-226            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-226          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-227            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-227           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-228            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-228           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-75                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-76                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-168                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-169             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-170                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-171                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-20                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-77                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-78                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-172                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-229            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-229          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-230            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-230           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-231            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-231           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-173                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-232            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-232          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-233            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-233           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-234            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-234           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-174                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-235            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-235          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-236            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-236           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-237            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-237           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-175       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-176                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-238            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-238          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-239            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-239           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-240            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-240           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-79                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-80                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-177                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-178             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-179                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-180                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-21                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-81                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-82                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-181                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-241            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-241          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-242            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-242           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-243            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-243           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-182                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-244            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-244          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-245            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-245           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-246            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-246           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-183                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-247            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-247          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-248            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-248           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-249            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-249           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-184       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-185                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-250            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-250          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-251            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-251           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-252            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-252           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-83                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-84                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-186                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-187             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-188                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-189                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-22                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-85                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-86                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-190                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-253            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-253          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-254            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-254           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-255            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-255           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-191                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-256            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-256          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-257            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-257           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-258            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-258           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-192                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-259            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-259          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-260            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-260           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-261            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-261           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-193       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-194                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-262            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-262          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-263            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-263           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-264            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-264           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-87                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-88                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-195                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-196             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-197                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-198                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-23                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-89                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-90                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-199                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-265            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-265          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-266            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-266           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-267            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-267           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-200                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-268            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-268          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-269            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-269           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-270            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-270           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-201                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-271            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-271          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-272            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-272           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-273            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-273           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-202       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-203                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-274            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-274          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-275            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-275           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-276            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-276           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-91                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-92                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-204                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-205             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-206                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-207                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-24                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-93                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-94                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-208                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-277            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-277          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-278            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-278           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-279            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-279           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-209                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-280            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-280          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-281            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-281           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-282            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-282           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-210                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-283            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-283          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-284            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-284           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-285            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-285           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-211       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-212                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-286            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-286          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-287            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-287           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-288            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-288           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-95                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-96                         [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-213                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-214             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-215                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-216                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-25                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-97                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-98                   [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-217                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-289            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-289          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-290            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-290           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-291            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-291           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-218                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-292            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-292          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-293            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-293           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-294            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-294           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-219                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-295            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-295          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-296            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-296           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-297            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-297           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-220       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-221                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-298            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-298          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-299            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-299           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-300            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-300           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-99                     [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-100                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-222                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-223             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-224                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-225                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-26                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-101                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-102                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-226                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-301            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-301          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-302            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-302           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-303            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-303           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-227                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-304            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-304          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-305            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-305           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-306            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-306           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-228                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-307            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-307          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-308            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-308           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-309            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-309           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-229       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-230                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-310            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-310          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-311            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-311           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-312            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-312           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-103                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-104                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-231                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-232             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-233                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-234                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-27                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-105                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-106                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-235                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-313            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-313          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-314            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-314           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-315            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-315           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-236                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-316            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-316          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-317            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-317           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-318            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-318           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-237                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-319            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-319          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-320            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-320           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-321            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-321           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-238       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-239                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-322            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-322          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-323            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-323           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-324            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-324           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-107                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-108                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-240                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-241             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-242                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-243                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-28                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-109                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-110                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-244                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-325            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-325          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-326            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-326           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-327            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-327           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-245                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-328            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-328          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-329            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-329           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-330            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-330           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-246                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-331            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-331          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-332            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-332           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-333            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-333           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-247       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-248                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-334            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-334          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-335            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-335           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-336            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-336           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-111                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-112                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-249                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-250             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-251                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-252                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-29                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-113                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-114                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-253                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-337            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-337          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-338            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-338           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-339            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-339           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-254                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-340            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-340          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-341            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-341           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-342            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-342           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-255                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-343            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-343          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-344            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-344           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-345            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-345           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-256       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-257                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-346            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-346          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-347            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-347           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-348            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-348           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-115                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-116                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-258                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-259             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-260                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-261                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-30                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-117                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-118                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-262                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-349            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-349          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-350            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-350           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-351            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-351           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-263                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-352            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-352          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-353            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-353           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-354            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-354           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-264                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-355            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-355          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-356            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-356           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-357            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-357           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-265       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-266                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-358            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-358          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-359            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-359           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-360            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-360           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-119                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-120                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-267                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-268             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-269                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-270                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-31                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-121                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-122                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-271                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-361            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-361          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-362            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-362           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-363            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-363           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-272                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-364            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-364          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-365            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-365           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-366            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-366           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-273                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-367            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-367          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-368            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-368           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-369            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-369           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-274       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-275                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-370            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-370          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-371            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-371           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-372            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-372           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-123                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-124                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-276                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-277             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-278                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-279                     [1, 86, 4096]                                      (45,088,768)
│    │    │    │    └─LlamaDecoderLayer: 5-32                     [1, 86, 4096]                                      --
│    │    │    │    │    └─LlamaRMSNorm: 6-125                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaAttention: 6-126                  [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-280                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-373            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-373          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-374            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-374           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-375            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-375           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-281                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-376            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-376          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-377            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-377           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-378            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-378           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─Linear: 7-282                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-379            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-379          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-380            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-380           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-381            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-381           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    │    └─LlamaRotaryEmbedding: 7-283       [1, 1, 86, 128]                                    --
│    │    │    │    │    │    └─Linear: 7-284                     [1, 86, 4096]                                      16,777,216
│    │    │    │    │    │    │    └─ModuleDict: 8-382            --                                                 --
│    │    │    │    │    │    │    │    └─Dropout: 9-382          [1, 86, 4096]                                      --
│    │    │    │    │    │    │    └─ModuleDict: 8-383            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-383           [1, 86, 16]                                        (65,536)
│    │    │    │    │    │    │    └─ModuleDict: 8-384            --                                                 --
│    │    │    │    │    │    │    │    └─Linear: 9-384           [1, 86, 4096]                                      (65,536)
│    │    │    │    │    └─LlamaRMSNorm: 6-127                    [1, 86, 4096]                                      (4,096)
│    │    │    │    │    └─LlamaMLP: 6-128                        [1, 86, 4096]                                      --
│    │    │    │    │    │    └─Linear: 7-285                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─SiLUActivation: 7-286             [1, 86, 11008]                                     --
│    │    │    │    │    │    └─Linear: 7-287                     [1, 86, 11008]                                     (45,088,768)
│    │    │    │    │    │    └─Linear: 7-288                     [1, 86, 4096]                                      (45,088,768)
│    │    │    └─LlamaRMSNorm: 4-3                                [1, 86, 4096]                                      (4,096)
│    │    └─Linear: 3-2                                           [1, 86, 32000]                                     (131,072,000)
=====================================================================================================================================================================
Total params: 6,755,192,832
Trainable params: 0
Non-trainable params: 6,755,192,832
Total mult-adds (G): 4.61
=====================================================================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 572.50
Params size (MB): 9215.42
Estimated Total Size (MB): 9787.92
=====================================================================================================================================================================
'''