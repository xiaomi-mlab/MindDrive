#!/bin/bash
# ================= Config =================
BASE_PORT=30000
BASE_TM_PORT=50000
BASE_ROUTES=data/test_opensource_routes/rollout_routes
CONFIG=adzoo/minddrive/configs/minddrive_qwen2_05B_lora_rollout.py
PLANNER_TYPE=traj
ALGO=minddrive_collect_ma_stage3
SAVE_PATH=./carla/rollout_routes_${ALGO}_${PLANNER_TYPE}
BASE_CHECKPOINT_ENDPOINT=collection_routes_test
DECOUPLE_SCRIPT="rl_projects/decode_rollout_dataset.py" 
DECOUPLE_OUTPUT="./carla/rollout_data/rollout_data_processed_${ALGO}_$(date +%Y%m%d)_pkl"
# ==========================
SPLIT_NUM=44

QUEUE_DIR="${BASE_ROUTES}_queue_${ALGO}"
if [ -d "$QUEUE_DIR" ]; then
    for file in "$QUEUE_DIR"/processing_*; do
        [ -e "$file" ] || continue
        
        filename=$(basename "$file")
        
        new_name=$(echo "$filename" | sed -E 's/processing_gpu[0-9]+_//')
        
        echo "Resetting interrupted task: $filename -> $new_name"
        mv "$file" "$QUEUE_DIR/$new_name"
    done
fi
# ====================================================

LOCK_FILE="./carla_collection_${ALGO}.lock"


if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p "$SAVE_PATH"
    echo -e "\033[32m Directory $SAVE_PATH created. \033[0m"
fi

NEED_GENERATE=true
if [ -d "$QUEUE_DIR" ]; then
    count=$(find "$QUEUE_DIR" -maxdepth 1 -name "*.xml" ! -name "processing_*" | wc -l)
    if [ "$count" -gt 0 ]; then
        echo -e "\033[32m Queue directory exists with $count tasks. Resuming... \033[0m"
        NEED_GENERATE=false
    else
        echo -e "\033[33m Queue directory empty or invalid. Regenerating... \033[0m"
        rm -rf "$QUEUE_DIR"
    fi
fi

if [ "$NEED_GENERATE" = true ]; then
    echo -e "****************************\033[33m Splitting XML \033[0m ****************************"
    mkdir -p "$QUEUE_DIR"
    
    python rl_projects/utils/split_xml.py $BASE_ROUTES $SPLIT_NUM $ALGO $PLANNER_TYPE
    
    echo -e "\033[32m Moving generated xml files to $QUEUE_DIR ... \033[0m"
    mv ${BASE_ROUTES}_*_${ALGO}_${PLANNER_TYPE}.xml "$QUEUE_DIR/"
    
    echo -e "\033[32m Tasks prepared in $QUEUE_DIR. \033[0m"
fi


run_worker() {
    local gpu_id=$1
    local port=$((BASE_PORT + gpu_id * 300))
    local tm_port=$((BASE_TM_PORT + gpu_id * 300))
    
    echo -e "\033[32m [GPU $gpu_id] Worker launched. \033[0m"
    
    while true; do

        task_path=""
        original_filename=""
        
        {
            flock -x 200 
            file_found=$(find "$QUEUE_DIR" -maxdepth 1 -name "*.xml" ! -name "processing_*" | head -n 1)
            
            if [ -n "$file_found" ]; then
                original_filename=$(basename "$file_found")
                task_path="${QUEUE_DIR}/processing_gpu${gpu_id}_${original_filename}"
                
                mv "$file_found" "$task_path"
            fi
        } 200>"$LOCK_FILE"

        if [ -z "$task_path" ]; then
            echo -e "\033[33m [GPU $gpu_id] No more tasks. Finishing. \033[0m"
            break
        fi

        clean_name=${original_filename%.xml}
        
        CHECKPOINT_JSON="${SAVE_PATH}/${BASE_CHECKPOINT_ENDPOINT}_${clean_name}.json"
        
        echo -e "\033[34m [GPU $gpu_id] Processing: $original_filename \033[0m"
        
        bash -e adzoo/minddrive/run_collection.sh \
            $port $tm_port "$task_path" "$CHECKPOINT_JSON" \
            "$SAVE_PATH" "$gpu_id" "$CONFIG" > "${task_path}.log" 2>&1
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo -e "\033[32m [GPU $gpu_id] Success: $original_filename \033[0m"
            rm "$task_path"
        else
            echo -e "\033[31m [GPU $gpu_id] FAILED: $original_filename. Log: ${task_path}.log \033[0m"
            
            {
                flock -x 200
                mv "$task_path" "${QUEUE_DIR}/${original_filename}"
                echo -e "\033[33m [GPU $gpu_id] Task ${original_filename} returned to queue. \033[0m"
            } 200>"$LOCK_FILE"
        
            sleep 2
        fi
    done
}

echo -e "***********************************************************************************"
echo -e "Starting Dynamic Load Balancing..."
echo -e "***********************************************************************************"

# for i in {0..7}; do
#     run_worker $i &
# done

# for debug
for i in 0; do
    run_worker $i &
done

wait

rm -f "$LOCK_FILE"
echo -e "\033[32m All Done. \033[0m"


# ================= Decouple =================
echo -e "***********************************************************************************"
echo -e "Starting Data Decoupling..."
echo -e "Source: $SAVE_PATH"
echo -e "Target: $DECOUPLE_OUTPUT"
echo -e "***********************************************************************************"

python "$DECOUPLE_SCRIPT" \
    --folder "$SAVE_PATH" \
    --output "$DECOUPLE_OUTPUT"

if [ $? -ne 0 ]; then
    echo -e "\033[31m Decoupling failed! Aborting training. \033[0m"
    exit 1
fi

echo -e "\033[32m Decoupling finished successfully. \033[0m"
