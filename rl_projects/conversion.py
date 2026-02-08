import torch
import os
import copy
from collections import OrderedDict

def process_weights(input_path, output_path):
    weights = torch.load(input_path, map_location='cpu')
    state_dict = weights.get('state_dict', weights)

    lm_head_keys = [k for k in state_dict if 'lm_head' in k]
    new_state_dict = OrderedDict()
    llm_keys, value_keys = [], []

    for key in state_dict:
        new_state_dict[key] = state_dict[key]

    for key in lm_head_keys:
        orig_key = key

        if key.startswith('lm_head.'):
            llm_key = key.replace('lm_head.', 'llm_model.', 1)
            value_key = key.replace('lm_head.', 'value_net.', 1)
        else:
            parts = key.split('.')
            try:
                lm_head_idx = parts.index('lm_head')
                parts[lm_head_idx] = 'llm_model'
                llm_key = '.'.join(parts)
                parts[lm_head_idx] = 'value_net'
                value_key = '.'.join(parts)
            except ValueError:
                llm_key = 'llm_model.' + key
                value_key = 'value_net.' + key

        new_state_dict[llm_key] = copy.deepcopy(state_dict[orig_key])
        llm_keys.append(llm_key)

        new_state_dict[value_key] = copy.deepcopy(state_dict[orig_key])
        value_keys.append(value_key)


    if hasattr(state_dict, '_metadata'):

        new_state_dict._metadata = copy.deepcopy(state_dict._metadata)
        for orig_key in lm_head_keys:
            if orig_key in state_dict._metadata:
                orig_meta = state_dict._metadata[orig_key]
                # llm_model key
                if orig_key.startswith('lm_head.'):
                    llm_key = orig_key.replace('lm_head.', 'llm_model.', 1)
                    value_key = orig_key.replace('lm_head.', 'value_net.', 1)
                else:
                    parts = orig_key.split('.')
                    try:
                        lm_head_idx = parts.index('lm_head')
                        parts_llm = parts.copy()
                        parts_llm[lm_head_idx] = 'llm_model'
                        llm_key = '.'.join(parts_llm)
                        parts_val = parts.copy()
                        parts_val[lm_head_idx] = 'value_net'
                        value_key = '.'.join(parts_val)
                    except ValueError:
                        llm_key = 'llm_model.' + orig_key
                        value_key = 'value_net.' + orig_key
                new_state_dict._metadata[llm_key] = copy.deepcopy(orig_meta)
                new_state_dict._metadata[value_key] = copy.deepcopy(orig_meta)

    # 复制 static._metadata
    if 'static' in weights:
        static = weights['static']
        new_static = OrderedDict(static) 
        if hasattr(static, '_metadata'):
            new_static._metadata = copy.deepcopy(static._metadata)
        if 'state_dict' in weights:
            weights['state_dict'] = new_state_dict
            weights['static'] = new_static
        else:
            weights = new_state_dict
            weights['static'] = new_static
    else:
        if 'state_dict' in weights:
            weights['state_dict'] = new_state_dict
        else:
            weights = new_state_dict

    torch.save(weights, output_path)

    return {
        "original_lm_head_keys": lm_head_keys,
        "new_llm_model_keys": llm_keys[:3],  
        "new_value_net_keys": value_keys[:3] 
    }

if __name__ == "__main__":
    input_path = "./work_dirs/orion_hisv2_tl_mm_ml_qwenv25_3b_pretrain_lora_stage3_meta_action_decouple_long/iter_11004.pth"
    output_path = os.path.splitext(input_path)[0] + "_v3.pth"
    result = process_weights(input_path, output_path)
    if result:
        print(f"原始 lm_head 键示例: {result['original_lm_head_keys'][0]}")
        print(f"新增 llm_model 键示例: {result['new_llm_model_keys'][0]}")
        print(f"新增 value_net 键示例: {result['new_value_net_keys'][0]}")