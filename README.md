<div align="center">
<h3> MindDrive: A Vision-Language-Action Model for Autonomous Driving Utilizing Language as Action in Online Reinforcement Learning </h3>

Haoyu Fu<sup>1\*</sup>, Diankun Zhang<sup>2\*</sup>, Zongchuang Zhao<sup>1</sup>, <br>Jianfeng Cui<sup>2</sup>, Hongwei Xie<sup>2†</sup>,  Bing Wang<sup>2</sup>, Guang Chen<sup>2</sup>, Dingkang Liang<sup>1†</sup>, Xiang Bai<sup>1</sup>

<sup>1</sup>  Huazhong University of Science & Technology, <sup>2</sup>  Xiaomi EV 

(\*) Equal contribution. (†) Project leader.

<a href="https://arxiv.org/abs/2512.13636"><img src='https://img.shields.io/badge/arXiv-MindDrive-red' alt='Paper PDF'></a>
<a href="https://xiaomi-mlab.github.io/MindDrive/"><img src='https://img.shields.io/badge/Project_Page-MindDrive-green' alt='Project Page'></a>
</div>


<!-- ## Introduction -->
## Abstract

Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. MindDrive achieves strong closed-loop performance on the challenging Bench2Drive benchmark, with a Driving Score (DS) of 78.04 and a Success Rate (SR) of 55.09\%. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.

## Overview
<div align="center">
<img src="assets/images/framework.png" width="1000">
</div>

## News
`[2026/02/08]` Minddrive code and dataset are now released!

`[2025/12/16]` [ArXiv](https://arxiv.org/abs/2512.13636) paper release.

## Currently Supported Features

- [x] MindDrive Inference Framework
- [x] Close-loop Evaluation
- [x] MindDrive Checkpoint
- [x] MindDrive Training Framework

## Getting Started

```
git clone https://github.com/xiaomi-mlab/MindDrive.git
cd ./MindDrive
conda create -n MindDrive python=3.8 -y
conda activate MindDrive
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -v -e .
pip install -r requirements.txt
```
Download and setup CARLA 0.9.15
```
mkdir /home/carla
cd /home/carla
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
tar -xvf CARLA_0.9.15.tar.gz
cd Import && wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
cd .. && bash ImportAssets.sh
export CARLA_ROOT=/home/carla
echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/MindDrive/lib/python3.8/site-packages/carla.pth
```
## Preparation
To help reproduce the results of Minddrive, we update the Chat-B2D dataset by incorporating meta-action VQA for each frame. The dataset is available at [here](https://huggingface.co/datasets/poleyzdk/Chat-B2D/resolve/main/ChatB2D-plus.zip?download=true).

We pretrain the [2D LLM weights](https://huggingface.co/poleyzdk/Minddrive/tree/main) and the [vision encoder + projector weights](https://huggingface.co/poleyzdk/Minddrive/tree/main) for the Qwen2-0.5B model, following the approach of Orion.

```
cd /path/to/Minddrive
mkdir ckpts
```

## Train
### Prepare data
Unzip the dataset:
```
unzip Chat-B2D-plus.zip -d data/
```
### Imitation Learning
Following Orion’s approach, this project uses a three-stage training pipeline (stage1, stage2, stage3). In the imitation learning stage we build a one-to-one mapping between language and trajectories.
```
./adzoo/minddrive/minddrive_dist_train.sh adzoo/minddrive/configs/minddrive_qwen2_05b_train_stage1.py $GPU
# or
./adzoo/minddrive/minddrive_dist_train.sh adzoo/minddrive/configs/minddrive_qwen2_05b_train_stage2(3).py $GPU
```
To save training time and GPU memory, we only train the LoRA of the action expert.
After Imitation Learning, we copy the action expert’s weights into the decision expert and the value net so they share the trained representations.

```
python rl_projects/convert_checkpoint.py
```

### Reinforcement Learning
Rollout (data collection):
```
bash adzoo/minddrive/minddrive_run_collection_multi.sh
```
The rollout script collects interaction data. The dataset is automatically decoupled and output to $DECOUPLE_OUTPUT.

RL training (PPO) example:
```
bash adzoo/minddrive/minddrive_run_mutil_train_ppo.sh 8 adzoo/minddrive/configs/minddrive_rl_ppo_train.py <imitation_weights_path> $DECOUPLE_OUTPUT/dataset_index.pkl
```

## Results and Checkpoints


### Orion and other baselines
| Method | L2 (m) 2s | Driving Score | Success Rate(%) | Config | Download | Eval Json|
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: |
| UniAD-Tiny |0.80 | 40.73 |  13.18 | [config](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad/adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_tiny_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/1psr7AKYHD7CitZ30Bz-9sA?pwd=1234 )| [Json](assets/results/UniAD-Tiny.json) |
| UniAD-Base |0.73 | 45.81  |  16.36 | [config](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad/adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/uniad_base_b2d.pth)/[Baidu Cloud](https://pan.baidu.com/s/11p9IUGqTax1f4W_qsdLCRw?pwd=1234) | [Json](assets/results/UniAD-Base.json) |
| VAD        |0.91 | 42.35  | 15.00 | [config](https://github.com/Thinklab-SJTU/Bench2DriveZoo/tree/uniad/vad/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py) | [Hugging Face](https://huggingface.co/rethinklab/Bench2DriveZoo/blob/main/vad_b2d_base.pth)/[Baidu Cloud](https://pan.baidu.com/s/1rK7Z_D-JsA7kBJmEUcMMyg?pwd=1234) | [Json](assets/results/VAD.json) |
| ORION-7B       |0.68 | 77.74  | 54.62 | [config](adzoo/orion/configs/orion_stage3.py) | [Hugging Face](https://huggingface.co/poleyzdk/Orion/blob/main/Orion.pth)| [Json](assets/results/ORION.json) |
MindDrive-0.5B   |0.69   | 78.04  | 55.09 | [config](adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py) | [Hugging Face](https://huggingface.co/poleyzdk/Minddrive/resolve/main/minddrive_rltrain.pth?download=true) | [Json](assets/results/minddrive.json) |


## Data Usage Statement

This project uses the following external resources:

- **Data**: We use the dataset provided in the Bench2Drive project (source: https://github.com/Thinklab-SJTU/Bench2Drive), which is licensed under the **CC BY-NC-ND 4.0** license.

- The authors confirm that the use of the above data in this project is strictly limited to academic research and has not involved any commercial activities.

## Citation
If this work is helpful for your research, please consider citing:

```
@article{fu2025minddrive,
  title={MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning},
  author={Haoyu Fu and Diankun Zhang and Zongchuang Zhao and Jianfeng Cui and Hongwei Xie and Bing Wang and Guang Chen and Dingkang Liang and Xiang Bai},
  journal={arXiv Preprint arXiv:2512.13636},  
  year={2025},
}
```
```
@inproceedings{fu2025orion,
  title={ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation},
  author={Haoyu Fu and Diankun Zhang and Zongchuang Zhao and Jianfeng Cui and Dingkang Liang and Chong Zhang and Dingyuan Zhang and Hongwei Xie and Bing Wang and Xiang Bai},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
