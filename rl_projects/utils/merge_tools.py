import os
import pickle
import numpy as np
from mmcv.runner.buffers import RolloutBuffer

def merge_multiple_rollout_buffers(buffer_paths, merged_path):
    buffers = []
    for p in buffer_paths:
        with open(p, 'rb') as f:
            buffers.append(pickle.load(f))
    
    total_size = sum(b.pos for b in buffers)
    first = buffers[0]
    merged_buffer = RolloutBuffer(
        buffer_size=total_size,
        device="auto",
        gae_lambda=first.gae_lambda,
        gamma=first.gamma,
        n_envs=first.n_envs,
        cache_obs=first.cache_obs
    )
    merged_buffer.pos = total_size
    merged_buffer.full = True

    array_attrs = [
        "actions", "rewards", "returns", 
        "episode_starts", "values", "ref_log_probs", 
        "advantages", "meta_action_info"
    ]
    for attr in array_attrs:
        merged = np.concatenate([getattr(b, attr)[:b.pos] for b in buffers], axis=0)
        setattr(merged_buffer, attr, merged)
    
    merged_buffer.generator_ready = False

    with open(merged_path, 'wb') as f:
        pickle.dump(merged_buffer, f)
    print(f"Merged buffer saved to {merged_path}")
    print(f"Final size: {total_size} timesteps")
    return merged_buffer

def get_all_pkl_files(root_dir):
    pkl_files = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        for fn in os.listdir(folder_path):
            fp = os.path.join(folder_path, fn)
            if fp.endswith('.pkl'):
                pkl_files.append(os.path.join(fp))
    return pkl_files

if __name__ == '__main__':
    root_dir = './carla/eval_bench2drive220_orion_collect_0724_5_traj'
    pkl_files = get_all_pkl_files(root_dir)
    if len(pkl_files) < 2:
        print("至少需要两个pkl文件才能合并")
        exit(1)
    merge_multiple_rollout_buffers(pkl_files, f"{root_dir}/merged_rollout_buffer.pkl")