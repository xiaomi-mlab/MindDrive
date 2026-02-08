GPUS=$1
CONFIG=$2
WEIGHT=$3
DATA=$4


# --- Environment Setup for Distributed Training ---
export RANK=${MLP_ROLE_INDEX:-0}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
export MASTER_PORT=${MLP_WORKER_0_PORT:-"29500"} # Use a default port
export WORLD_SIZE=${MLP_WORKER_NUM:-1}

# --- CARLA and Project Specific Environment Variables ---
export PYTHONPATH="${PWD}"
export PYTHONPATH="/home/carla/PythonAPI:${PYTHONPATH}"
export PYTHONPATH="/home/carla/PythonAPI/carla:${PYTHONPATH}"
export PYTHONPATH="rl_projects/:${PYTHONPATH}"
export PYTHONPATH="rl_projects/scenario_runner:${PYTHONPATH}"
export TORCH_DISTRIBUTED_DEBUG=INFO
export CARLA_ROOT="/home/carla"
export CARLA_SERVER="/home/carla/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="rl_projects/scenario_runner"
export TORCH_NCCL_ENABLE_TIMING=1

# --- Workspace and Logging Setup ---
# Create a work directory based on the config file name
WORK_DIR=$(echo ${CONFIG%.*} | sed -e "s,configs,work_dirs,g")
mkdir -p ${WORK_DIR}/logs

# --- Print Environment Information ---
echo "=========================================="
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS per node: ${GPUS}"
echo "CONFIG_FILE: ${CONFIG}"
echo "WORK_DIR: ${WORK_DIR}"
echo "WEIGHT: ${WEIGHT}"
echo "DATA: ${DATA}"

echo "=========================================="

# --- Launch Command ---
# Use torch.distributed.launch for distributed training
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=${WORLD_SIZE} \
    --node_rank=$RANK \
    --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    adzoo/minddrive/train.py \
    $CONFIG \
    --load_from ${WEIGHT} \
    --rollout_data  ${DATA} \
    --launcher pytorch \
    2>&1 | tee "${WORK_DIR}/logs/std_out_rank_${RANK}.log"