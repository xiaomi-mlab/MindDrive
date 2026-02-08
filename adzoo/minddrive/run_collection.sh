#!/bin/bash
export PYTHONPATH="${PWD}"
export PYTHONPATH="/home/carla/PythonAPI:${PYTHONPATH}"
export PYTHONPATH="/home/carla/PythonAPI/carla:${PYTHONPATH}"
export PYTHONPATH="rl_projects/:${PYTHONPATH}"
export PYTHONPATH="rl_projects/scenario_runner:${PYTHONPATH}"

export TORCH_DISTRIBUTED_DEBUG="INFO"
export CARLA_ROOT="/home/carla"
export CARLA_SERVER="/home/carla/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="rl_projects/scenario_runner"
export DEBUG_SHOW_PRED="1"

export PORT=$1
export TM_PORT=$2
export REPETITIONS=5 # multiple evaluation runs

# TCP evaluation
export ROUTES=$3
export CHECKPOINT_ENDPOINT=$4
export SAVE_PATH=$5
export GPU_RANK=$6
export CONFIG=$7
echo -e "GPU_RANK: $GPU_RANK"
CUDA_VISIBLE_DEVICES=${GPU_RANK} python adzoo/minddrive/rollout.py \
    ${CONFIG} \
    --routes=${ROUTES} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --port=${PORT} \
    --traffic_manager_port=${TM_PORT} \
    --repetitions=${REPETITIONS} \
    --resume \
    --use_carla \