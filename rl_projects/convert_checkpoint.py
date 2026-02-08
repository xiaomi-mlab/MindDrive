import torch
import copy
import os

def transfer_action_to_decision(input_path, output_path):
    # Load the trained weights
    print(f"Loading weights from {input_path}...")
    weights = torch.load(input_path, map_location="cpu")
    state_dict = weights.get('state_dict', weights)
    new_state_dict = state_dict.copy()
    
    source_keyword = "action_expert"
    target_keyword = "decision_expert"
    
    transfer_count = 0
    value_net_count = 0
    
    print("Starting processing keys...")
    
    for key in list(state_dict.keys()):
        
        if source_keyword in key:
            target_key = key.replace(source_keyword, target_keyword)
            new_state_dict[target_key] = state_dict[key].clone()
            transfer_count += 1

        if "lm_head" in key:
            if source_keyword in key and ("lora_A" in key or "lora_B" in key):
                target_value_key = key.replace("lm_head", "value_net").replace(source_keyword, "default")
                print(f"[Value Net LoRA] {key} -> {target_value_key}")
            elif target_keyword in key:
                    continue
            else:
                target_value_key = key.replace("lm_head", "value_net")
            new_state_dict[target_value_key] = state_dict[key].clone()
            value_net_count += 1

    if hasattr(state_dict, '_metadata'):    
        print("Processing _metadata for value_net...")
        new_metadata = copy.deepcopy(state_dict._metadata)
        
        for key in list(state_dict._metadata.keys()):
            if "lm_head" in key:
                if source_keyword in key:
                    value_net_meta_key = key.replace("lm_head", "value_net").replace(source_keyword, "default")
                elif target_keyword in key:
                    continue
                else:
                    value_net_meta_key = key.replace("lm_head", "value_net")
 
                new_metadata[value_net_meta_key] = copy.deepcopy(state_dict._metadata[key])
        
        new_state_dict._metadata = new_metadata

    if 'state_dict' in weights:
        weights['state_dict'] = new_state_dict
    else:
        weights = new_state_dict

    print(f"\nSummary:")
    print(f"- Transferred {transfer_count} layers (action -> decision).")
    print(f"- Created {value_net_count} value_net layers (lm_head copy).")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving new weights to {output_path}...")
    torch.save(weights, output_path)
    print("Done.")

if __name__ == "__main__":
    input_model_path = "work_dirs/minddrive_qwen2_05b_train_stage3/iter_11004.pth"
    output_model_path = "work_dirs/minddrive_qwen2_05b_train_stage3/iter_11004_Minddrive.pth"
    
    transfer_action_to_decision(input_model_path, output_model_path)