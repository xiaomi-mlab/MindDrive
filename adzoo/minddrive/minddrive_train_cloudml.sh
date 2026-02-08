GPUS=$1
CONFIG=$2
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
T=`date +%m%d%H%M`
export RANK=$MLP_ROLE_INDEX
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export WORLD_SIZE=$MLP_WORKER_NUM

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}
export TORCH_NCCL_ENABLE_TIMING=1

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "GPUS: ${GPUS}"
echo "WORK_DIR: ${WORK_DIR}"
mkdir -p ${WORK_DIR}
if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch --nnodes=${WORLD_SIZE} \
    --node_rank=$RANK --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:4} \
    2>&1 | tee "${WORK_DIR}/std_out_rank_${RANK}.log"