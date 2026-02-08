#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM, \
                            Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, use_gen_token=False, use_critical_qa=False,use_meta_action=False, value_net = False):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQwen2Model(config)
        self.hidden_size = config.hidden_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.pretraining_tp = config.pretraining_tp

        number_tokens = [
                10,
                12,
                15,
                13,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ]  # +-0.123456789
        if use_gen_token and not use_meta_action:
            weighted_mask = torch.ones(self.config.tokenizer_vocab_size + 2) # TODO：这里直接+1，适应于新token
            weighted_mask[number_tokens] = 1.0
        elif use_gen_token and use_meta_action:
            weighted_mask = torch.ones(self.config.tokenizer_vocab_size + 14) # TODO：这里直接+1，适应于新token
            weighted_mask[number_tokens] = 1.0
        else:
            weighted_mask = torch.ones(self.config.vocab_size)
            weighted_mask[number_tokens] = 3.0
        if use_critical_qa:
            weighted_mask[number_tokens] = 3.0
        self.register_buffer("weighted_mask", weighted_mask)


        self.use_gen_token = use_gen_token
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward_rl_value(
        self,
        meta_action_info: dict,
    ):

        inputs_embeds = meta_action_info['inputs_embeds'] # (16, 575, 896)
        new_input_ids = meta_action_info['new_input_ids'] # (16, 575)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)

        outputs = self.model(
            input_ids= None, # None
            attention_mask=attention_mask, # (1, 588)
            position_ids=None, # None
            past_key_values=None, # None
            inputs_embeds=inputs_embeds, # (1, 588, 4096)
            use_cache=False, # False
            output_attentions=False, # False
            output_hidden_states=False, # False
            return_dict=True, # True
        )

        hidden_states = outputs[0] # (7, 581, 896)
        
        loc_positions_list = []
        for new_id in new_input_ids:
            loc_positions = torch.zeros_like(new_id).to(torch.bool)
            for token_id in self.config.meta_action_token_idx[:7]:
                if token_id in new_id:
                    loc_positions = torch.logical_or(loc_positions, new_id == token_id)
            loc_positions_list.append(loc_positions)

        # prev[i] = loc[i+1]
        prev_loc_positions_list = []
        for loc_positions in loc_positions_list:
            prev = torch.zeros_like(loc_positions)
            prev[:-1] = loc_positions[1:] 
            prev_loc_positions_list.append(prev)
        loc_positions = torch.stack(prev_loc_positions_list,dim=0)
        selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]

        return selected_hidden_states
        
    def forward_rl(
        self,
        meta_action_info: dict,
    ):
        inputs_embeds = meta_action_info['inputs_embeds'] # (16, 575, 896)
        new_input_ids = meta_action_info['new_input_ids'] # (16, 575)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)

        outputs = self.model(
            input_ids= None, # None
            attention_mask=attention_mask, # (1, 588)
            position_ids=None, # None
            past_key_values=None, # None
            inputs_embeds=inputs_embeds, # (1, 588, 4096)
            use_cache=False, # False
            output_attentions=False, # False
            output_hidden_states=False, # False
            return_dict=True, # True
        )
        hidden_states = outputs[0]
        output_logits = self.lm_head(hidden_states)
        
        last_token_logits = output_logits[:, -2, :]  # shape: (batch, vocab_size) 倒数第二个才是，QA里有<meta-action>
        ma = torch.stack([torch.tensor(ma) for ma in self.config.meta_action_token_idx[:7]])
        ma = ma.repeat(last_token_logits.size(0), 1)
        ma_logits = last_token_logits.gather(
            dim=-1,
            index=ma.to(last_token_logits.device)
        )  # shape: (batch, num_meta_action)
        action_log_probs_normalized = F.log_softmax(ma_logits, dim=-1)
        log_probs = F.log_softmax(output_logits[:, :-1, :], dim=-1) # (1, 583, 151680)
        log_probs_labels = log_probs.gather(dim=-1, index=torch.clamp(new_input_ids[:, 1:], 0).unsqueeze(-1))
        return action_log_probs_normalized, log_probs_labels
   

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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_waypoints: Optional[bool] = False,
        return_ego_feature: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        else:
            new_input_ids = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids, # None
            attention_mask=attention_mask, # (1, 588)
            position_ids=position_ids, # None
            past_key_values=past_key_values, # None
            inputs_embeds=inputs_embeds, # (1, 588, 4096)
            use_cache=use_cache, # False
            output_attentions=output_attentions, # False
            output_hidden_states=output_hidden_states, # False
            return_dict=return_dict, # True
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]

        if return_waypoints:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = ( (new_input_ids == self.config.waypoint_token_idx))
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
                waypoint = self.waypoint_decoder(selected_hidden_states)
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            break
                    loc_positions = ( (new_id == token_id))
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list,dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
                waypoint = self.waypoint_decoder(selected_hidden_states)
        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = ( (new_input_ids == self.config.waypoint_token_idx))
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list,dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
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
            loss = torch.nan_to_num(loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # 使用特殊token：如果不加特殊token不参与训练，我加了特殊token，参与训链了，然后这个类型不太对，应该是torch.float32
        # self.model.embed_tokens = self.model.embed_tokens.to(torch.float32)
        # self.model.to(torch.float32)
        if return_waypoints:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), waypoint
        elif return_ego_feature:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), selected_hidden_states
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
       

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        output_ids = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        output_embeds = self.get_model().embed_tokens(output_ids)

        return output_ids, inputs_embeds, output_embeds, new_input_ids # inputs_embeds,(1, 575, 896)

    @torch.no_grad()
    def generate_test(
            self,
            meta_action_info: dict,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            **kwargs,
        ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        # if images is not None:
        #     (
        #         inputs,
        #         position_ids,
        #         attention_mask,
        #         _,
        #         inputs_embeds,
        #         _,
        #         new_input_ids
        #     ) = self.prepare_inputs_labels_for_multimodal(
        #         inputs,
        #         position_ids,
        #         attention_mask,
        #         None,
        #         None,
        #         images,
        #         image_sizes=image_sizes
        #     )
        # else:
        #     inputs_embeds = self.get_model().embed_tokens(inputs)
        inputs_embeds = meta_action_info['inputs_embeds']
        attention_mask=None
        position_ids = None
        output_ids = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        output_embeds = self.get_model().embed_tokens(output_ids)

        return output_ids, inputs_embeds, output_embeds # inputs_embeds,(1, 575, 896)

    @torch.no_grad()
    def inference_waypoints(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        return_ego_feature = False,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        return_dict = self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values= None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]

        if return_ego_feature:
            if not isinstance(self.config.waypoint_token_idx, list):
                loc_positions = ( (new_input_ids == self.config.waypoint_token_idx))
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            else:
                loc_positions_list = []
                for new_id in new_input_ids:
                    loc_positions = torch.zeros_like(new_id).to(torch.bool)
                    for token_id in self.config.waypoint_token_idx:
                        if token_id in new_id:
                            loc_positions = torch.logical_or(loc_positions, new_id == token_id)
                    loc_positions_list.append(loc_positions)
                loc_positions = torch.stack(loc_positions_list,dim=0)
                selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            return selected_hidden_states
        if not isinstance(self.config.waypoint_token_idx, list):
            loc_positions = ( (new_input_ids == self.config.waypoint_token_idx))
            selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            waypoint = self.waypoint_decoder(selected_hidden_states)
        else:
            loc_positions_list = []
            for new_id in new_input_ids:
                for token_id in self.config.waypoint_token_idx:
                    if token_id in new_id:
                        break
                loc_positions = ( (new_id == token_id))
                loc_positions_list.append(loc_positions)
            loc_positions = torch.stack(loc_positions_list,dim=0)
            selected_hidden_states = hidden_states[loc_positions.to(device = hidden_states.device)]
            waypoint = self.waypoint_decoder(selected_hidden_states)
        
        return waypoint
        
    @torch.no_grad()
    def inference_action_distribution(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_input_ids
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        output_attentions = self.config.output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
        )
        return_dict = self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values= None,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # find 2d position  self.model.to(torch.float32)
        hidden_states = outputs[0]
        output_logits = self.lm_head(hidden_states)
        
        # log_probs = F.log_softmax(output_logits[:, :-1, :], dim=-1) # (1, 583, 151680)

        # action_log_probs_language = log_probs[:, -1] # 取出最后一维度
        # ma = torch.stack([torch.tensor(ma) for ma in self.config.meta_action_token_idx[:7]])
        # # log_probs_labels = action_log_probs.gather(dim=-1, index=torch.clamp(new_input_ids[:, 1:], 0).unsqueeze(-1))
        # action_log_probs = action_log_probs_language.gather(dim=-1, index=ma.to(action_log_probs_language.device).unsqueeze(0))
        # action_log_probs_normalized = F.log_softmax(action_log_probs, dim=-1)
        # return action_log_probs_normalized, inputs_embeds, new_input_ids
        # 不全局
        last_token_logits = output_logits[:, -2, :]  # shape: (batch, vocab_size) 倒数第二个才是，QA里有<meta-action>
        ma = torch.stack([torch.tensor(ma) for ma in self.config.meta_action_token_idx[:7]])
        ma_logits = last_token_logits.gather(
            dim=-1,
            index=ma.to(last_token_logits.device).unsqueeze(0)
        )  # shape: (batch, num_meta_action)

        action_log_probs_normalized = F.log_softmax(ma_logits, dim=-1)

        return action_log_probs_normalized, inputs_embeds, new_input_ids

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaQwen2Model)

def add_special_token(special_token_list, tokenizer, model):
    # 给新的token添加索引并用大模型的embeding的平均值来初始化token的embeding
    num_new_tokens = tokenizer.add_tokens(special_token_list, special_tokens = True)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
