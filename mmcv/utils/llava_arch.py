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


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
    ):
        
        if  image_features is None or input_ids.shape[1] == 1:
            # if past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
            #     target_shape = past_key_values[-1][-1].shape[-2] + 1
            #     attention_mask = torch.cat((attention_mask, torch.ones(
            #         (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
            #         dtype=attention_mask.dtype,
            #         device=attention_mask.device
            #     )), dim=1)
            #     position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        
        if isinstance(image_features,list):
            temp_image_features = []
            for b_id in range(len(image_features[0])):
                for img_id in range(len(image_features)):
                    temp_image_features.append(image_features[img_id][b_id])
            image_features = temp_image_features
        else:
            image_features = image_features.reshape(image_features.shape[0], -1, self.hidden_size).to(dtype=self.dtype) # (B, 513, 4096)

        # TODO: image start / end is not implemented here to support pretraining.
        # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        #     raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool() # (B, 76)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) # (76,)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_input_ids = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids): # 遍历batch samples
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # 只有一个-200，位置在index 35，估计对应的是句子中image的占位
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] # [-1, 35, 76]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # 以image token位置为分界，分割出来句子块
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # [(35,), (40,)]，分块input ids
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]]) # [(35,), (40,)]，分块labels
            split_sizes = [x.shape[0] for x in cur_labels_noim] # [35, 40]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim).to(image_features.device)) # (75,) -> (75, 4096)，得到单词embedding
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # [(35, 4096), (40, 4096)]，分成分块单词embedding
            
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_input_ids = []

            for i in range(num_images + 1): # 遍历单词分块，在合适位置，embedding给append入image feature，label给append入IGNORE_INDEX，input id给append入IMAGE_TOKEN_INDEX，得到完整句子组成
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_input_ids.append(cur_input_ids_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_input_ids = torch.cat(cur_new_input_ids)
            # 组成batch
            new_input_embeds.append(cur_new_input_embeds) # [(588, 4096)], 588 = 35+513+40
            new_labels.append(cur_new_labels) # [(588,)]
            new_input_ids.append(cur_new_input_ids) # [(588,)]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds) # batch内samples的最大长度，这里想把整个batch都整到一起
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_inputs_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]

            #padding
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                new_inputs_ids_padded[i, :cur_len] = cur_new_input_ids
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_inputs_ids_padded
