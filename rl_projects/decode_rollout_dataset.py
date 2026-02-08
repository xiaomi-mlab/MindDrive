import os
import pickle
import argparse
import numpy as np


def find_pkl_files(root_dir):
    if not os.path.exists(root_dir):
        print(f"Error: Path does not exist: {root_dir}")
        return []

    return [os.path.join(root, f) 
            for root, _, files in os.walk(root_dir) 
            for f in files if f.endswith('.pkl')]


import os
import pickle
import numpy as np
import json
from tqdm import tqdm

def process_pkl_files(pkl_files, output_dir):
    """Decode .pkl files and save step data as .npz."""
    os.makedirs(output_dir, exist_ok=True)
    attrs = [
        "actions", "rewards", "returns", "episode_starts", 
        "values", "ref_log_probs", "advantages", "meta_action_info"
    ]
    step_idx, all_steps = 0, []

    for pkl_path in tqdm(pkl_files, desc="Processing pkl files"):
        try:
            with open(pkl_path, 'rb') as f:
                buffer = pickle.load(f)

            pos_value = getattr(buffer, 'pos', 0)
            print(f"Processing {pkl_path} with pos={pos_value}")
            
            if pos_value == 0:
                print(f"Skipping {pkl_path}: Invalid or empty buffer.")
                continue

            # Process each step in buffer
            for i in range(pos_value):
                step_data = {attr: getattr(buffer, attr)[i] for attr in attrs if hasattr(buffer, attr)}
                
                file_name = f"step_{step_idx:08d}.npz"
                step_path = os.path.join(output_dir, file_name)
                
                np.savez(step_path, **step_data)
                all_steps.append(step_path)
                step_idx += 1

        except _pickle.UnpicklingError as ue:
            print(f"UnpicklingError in {pkl_path}: {ue}")
        except AttributeError as ae:
            print(f"AttributeError in {pkl_path}: {ae}")
        except Exception as e:
            print(f"Error processing {pkl_path}: {e}")

    # Save the dataset index as a PKL file instead of JSON
    index_path = os.path.join(output_dir, "dataset_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(all_steps, f)

    print(f"Processed {step_idx} steps. Index saved to {index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Input folder containing .pkl files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save processed data.')
    args = parser.parse_args()

    pkl_files = find_pkl_files(args.folder)
    if not pkl_files:
        print("No .pkl files found. Exiting.")
        return

    print(f"{len(pkl_files)} files to process.")
    process_pkl_files(pkl_files, args.output)


if __name__ == '__main__':
    main()