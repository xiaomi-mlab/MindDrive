#!/usr/bin/env bash
GPUS=$1
CONFIG=$2
CHECKPOINT=$3

export RANK=$MLP_ROLE_INDEX
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export WORLD_SIZE=$MLP_WORKER_NUM
RANK=${RANK:=0}
MASTER_ADDR=${MASTER_ADDR:='127.0.0.1'}
MASTER_PORT=${MASTER_PORT:='23456'}


PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python  -m torch.distributed.launch --nnodes=${WORLD_SIZE} \
    --node_rank=$RANK --nproc_per_node=$GPUS \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5} --eval bbox \
    