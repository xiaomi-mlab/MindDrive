import os
import pickle
import argparse
import numpy as np

def get_all_pkl_files(root_dir):
    pkl_files = []
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fn in os.listdir(folder_path):
            fp = os.path.join(folder_path, fn)
            if fp.endswith('.pkl'):
                pkl_files.append(fp)
    return pkl_files

def decode_pkl_to_step_npzs(pkl_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    array_attrs = [
        "actions", "rewards", "returns", 
        "episode_starts", "values", "ref_log_probs", 
        "advantages", "meta_action_info"
    ]
    # step计数器
    step_idx = 0
    all_step_paths = []

    for pkl_path in pkl_files:
        print(f"Processing {pkl_path}")
        with open(pkl_path, 'rb') as f:
            buffer = pickle.load(f)
        attrs = array_attrs 
        num_steps = buffer.pos
        for i in range(num_steps):
            data = {}
            for attr in attrs:
                arr = getattr(buffer, attr)
                if isinstance(arr, (np.ndarray, list)):
                    data[attr] = arr[i]
                else:
                    data[attr] = arr[i]
            step_npz_path = os.path.join(output_dir, f"step_{step_idx:08d}.npz")
            np.savez(step_npz_path, **data)
            all_step_paths.append(step_npz_path)
            step_idx += 1

    print(f"全部step已保存到 {output_dir}, 共{step_idx}步.")
    index_path = os.path.join(output_dir, "dataset_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(all_step_paths, f)
    print(f"数据集路径索引已保存到 {index_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True, help='pkl所在文件夹')
    parser.add_argument('-o', '--output', default='./carla/rollout_data/eval_bench2drive220_orion_collect_failed_routes_rollout_2_stage3_kl_traj', help='输出数据集文件夹名')
    args = parser.parse_args()

    pkl_files = get_all_pkl_files(args.folder)
    if len(pkl_files) == 0:
        print("没有找到pkl文件")
        exit(1)
    decode_pkl_to_step_npzs(pkl_files, args.output)