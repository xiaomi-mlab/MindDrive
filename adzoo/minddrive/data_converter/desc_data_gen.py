from PIL import Image
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import torch
from tqdm import tqdm
import json
from os import path as osp
from PIL import Image, ImageOps
import base64
from io import BytesIO
import re
import argparse
import pickle
import os
from pathlib import Path
import time 
# qwen7b = '../../../zzc/weight/Qwen2-VL-7B-Instruct/'
qwen72b = 'weight/Qwen2-VL-72B-Instruct/'
MODEL_PATH = qwen72b
BATCH_SIZE = 10
# IMAGE_PATH = '/fusion-algo-lidar-secret-nas/interns/fhy/project/Qwen2-VL/front.jpeg'
# VIDEO_PATH = '/path/to/video.mp4'
data_root = './data/bench2drive'

def replace_newlines_in_json_string(s):
    pattern = re.compile(r'\"(.*?)\"', re.DOTALL)
    
    def replace_newline(match):
        return match.group(0).replace('\n', '\\n')
    
    replaced_string = re.sub(pattern, replace_newline, s)
    return replaced_string

def preprocess_images(image_paths_or_images, layout='horizontal', flip=False):
    images = [Image.open(x) if isinstance(x, str) else x for x in image_paths_or_images]

    if flip:
        images = [ImageOps.mirror(image) for image in images]

    if layout == 'horizontal':
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
    elif layout == 'vertical':
        widths, heights = zip(*(i.size for i in images))
        max_width = max(widths)
        total_height = sum(heights)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
    return new_im

def create_combined_image(front_image_paths, back_image_paths):
    front_image = preprocess_images(front_image_paths)
    back_image = preprocess_images(back_image_paths, flip=True)
    return front_image, back_image

def main(info_file, output_dir, begin):
    
    # 加载模型
    llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=8,
            limit_mm_per_prompt={'image': 2},
            # cpu_offload_gb = 100
        )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    sampling_params = SamplingParams(
    temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=2560,
    stop_token_ids=[],
    )
    # temperature：控制采样随机性的浮点数。较低的值使模型更具确定性，而较高的值使模型更随机。零值意味着贪婪采样。top_p：控制考虑的顶部标记的累积概率的浮点数。必须在 (0, 1] 范围内。设置为 1 以考虑所有标记。
    # repetition_penalty：一个浮点数，根据新标记是否出现在提示和生成的文本中来惩罚新标记。值大于 1 鼓励模型使用新标记，而值小于 1 鼓励模型重复标记
    # max_tokens：每个输出序列生成的最大标记数量
    
    key_infos = pickle.load(open(info_file, 'rb'))

    for i in tqdm(range(begin, len(key_infos), BATCH_SIZE)):
        # 加载批数据
        batch = key_infos[i:i+BATCH_SIZE]     
        all_llm_inputs = []
        all_savepath = []

        for data in batch:
            image_paths = {}
            for sensor_type, cam_info in data['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths[sensor_type] = os.path.join(data_root, cam_info['data_path'])

            front_image_paths = [image_paths['CAM_FRONT_LEFT'], image_paths['CAM_FRONT'], image_paths['CAM_FRONT_RIGHT']]
            back_image_paths =  [image_paths['CAM_BACK_LEFT'], image_paths['CAM_BACK'], image_paths['CAM_BACK_RIGHT']]
            front_image, back_image = create_combined_image(front_image_paths, back_image_paths)

            front_image = front_image.resize((1540, 532)) # 55* 19 =1045
            back_image = back_image.resize((1540, 532))
            sys_prompt = f"""Given two panoramic images that encapsulates the surroundings of a vehicle in a 360-degree view, your task is to analyze and interpret the current driving behavior and the associated driving scene.
            Your task is divided into two parts:
            1. Summarize the driving scenario in a paragraph.
            - In this task, you should provide a detailed description of the driving scene. For example, specify the road condition, \
            noting any particular settings (parking lot, intersection, roundabout), traffic elements (pedestrain, vehicle, traffic sign/light), time of the day and weather.

            2. Analyze the driving action.
            - The task is to use the given image to shortly explain the driving intentions, assuming you are driving in a real scene.
            - You should understand the provided image, first identify the proper driving decision/intension. \
            Then based on your background knowledge to reason what the driver should be particularly mindful of in this scenario and list them in the point form.
            - Do not directly copy the provided planning infomation; instead, make the action description sound more natural.

            In both tasks:
            - Do not mention the "first/second image", ""front-left/rear-center view" respectively describes xxx. Instead, replace it with what is present at specific vehicle positions (front, back, left, right, etc.). \
            Always answer as if you are directly in the driving scene.
            - When describing the traffic elements, please specify their location or appearance characteristics to make them more distinguishable. \
            Do not merely mention generic traffic rules; integrate the information from the image.
            - Each panoramic image is a composite of three smaller images. The first image depicts scenes to the left-front, directly in front, and right-front of the vehicle. \
            The second image, displays views of the left-rear, directly behind, and right-rear of the same vehicle.
            - Answer based only on the content determined in the image, and do not speculate on uncertain content.

            You should refer to the following example and format the results like {{"description": "xxx", "action": "xxx"}}:

            {{
                "description": "The scene captures a moment of urban life framed by a red traffic light in mid-transition. To the right, a pedestrian crossing, marked by bright white zebra stripes, lies momentarily empty, waiting for the signal to change. \
            Directly ahead, a lineup of vehicles—a mix of sedans, a motorcycle, and a delivery van—pauses obediently at the red light, their headlights beginning to flicker on against the dimming light. \
            On the left, the sidewalk bustles with people of all ages, indicating a neighborhood that thrives on its mix of residential and commercial energy. \
            Behind this foreground of orderly traffic and pedestrian movement, the cityscape reveals a patchwork of modern and older buildings. A parked truck is behind us, with a construction worker standing beside it.",
                "action": "In this scenario, the vehicle should move slowly and make a right lane change. 
            - The decision to change lanes is influenced by the need to overtake the stop bus in front of the vehicle. 
            - There are no traffic behind the vehicle and ensure a gap large enough for a safe lane change. 
            - Pedestrians are visible on the sidewalk to the right, it is necessary to observe their movements when changing lanes."
            }},
            """
            # front_image.save(f'test_image/front_image_{i}.jpg')
            # back_image.save(f'test_image/back_image_{i}.jpg')
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": front_image,
                        },
                        {
                            "type": "text",
                            "text": "The first image depicts scenes to the left-front, directly in front, and right-front of the vehicle.",
                        },
                        {
                            "type": "image",
                            "image": back_image,
                        },
                        {
                            "type": "text",
                            "text": "The second image, displays views of the left-rear, directly behind, and right-rear of the vehicle.",
                        },
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            mm_data = {}
            if image_inputs is not None:
                mm_data['image'] = image_inputs

            llm_inputs = {
                'prompt': prompt,
                'multi_modal_data': mm_data,
            }

            all_llm_inputs.append(llm_inputs)
            # 输出路径
            image_path = Path(image_paths['CAM_FRONT'])
            # 获取JSON文件所在目录
            json_directory = image_path.parent.parent.parent.stem # 上溯
            # 构造JSON文件路径
            save_path = output_dir
            save_path = save_path + '/'+ str(json_directory)+'/'+'conv'+'/'
            json_path = f'{image_path.stem}.json'
            json_path_str = str(json_path)
            output_file_path = save_path + json_path_str # 保存路径
            all_savepath.append(output_file_path)

        llmoutputs = llm.generate(all_llm_inputs, sampling_params=sampling_params)
        time.sleep(1)
        # 保存一个batch的数据
        for index in range(BATCH_SIZE):
            
            generated_text = llmoutputs[index].outputs[0].text
            # 输出到文件夹中
            if not os.path.exists(all_savepath[index]):   
                os.makedirs(osp.dirname(all_savepath[index]), exist_ok=True)
                # 将字典保存为JSON文件
                result = replace_newlines_in_json_string(generated_text)
                with open(all_savepath[index], 'w') as f:
                    json.dump(result, f, indent=4)
                # print(all_savepath[index])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process NuScenes data.")
    parser.add_argument('--info_file', type=str, default='./data/infos/b2d_infos_train.pkl', help='Path to the info file (e.g., nuscenes2d_ego_temporal_infos_train.pkl).')
    # parser.add_argument('--desc_path', type=str, default='./desc/train/', help='Path to the description files directory.')
    parser.add_argument('--output_dir', type=str, default='./data/train_desc', help='Directory to save the output JSON files.')
    parser.add_argument('--begin', type=int, default=0, help='Number of Begin')

    args = parser.parse_args()
    main(args.info_file, args.output_dir, args.begin)
