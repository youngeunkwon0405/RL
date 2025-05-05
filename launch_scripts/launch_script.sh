#!/bin/bash

COMMAND_FILE="$1"
ADD_ARG="$2"
# Source the configuration file provided as the first argument
LOGS_DIR=/lustre/fsw/portfolios/coreai/users/${USER}/slurm_logs
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/hf_cache
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/coreai/projects/coreai_nvfm_llm/datasets/hf_data/
# Activate secrets file that includes HF tokens, wandb, etc...
source "/lustre/fsw/portfolios/coreai/users/${USER}/secrets"

# Job params
N_RUNS=1
NODES=2
GPUS_PER_NODE=8
JOB_NAME="RL_MATH"
IMAGE_PATH='gitlab-master.nvidia.com/deci/research/lit-llama/puzzle_amnon:latest'
# Function to submit a job and return the job ID
if [ "$CLUSTER_NAME" = "CS_CW_DFW" ] || [ "$CLUSTER_NAME" = "CW_PDX_CS_001" ]; then
  PARTITION='batch_short' #grizzly,polar,polar2,polar3,polar4
else
  PARTITION='grizzly,polar,polar2,polar3,polar4'
fi
#PARTITION='backfill'
submit() {
    local dep=$1
    local common_args="submit_job \
        --time=${JOB_HOURS:-"0"} \
        --gpu=${GPUS_PER_NODE} \
        --nodes=${NODES} \
        --partition=${PARTITION} \
        --exclusive \
        --skip_image_check \
        --image=${IMAGE_PATH} \
        --name=${JOB_NAME} \
        --logroot=${LOGS_DIR}/${JOB_NAME} \
        --mounts /lustre \
        --email_mode='always' \
        --notify_on_start  \
        --notification_method='slack' \
        --setenv HF_HOME=${HF_HOME},WANDB_API_KEY=${WANDB_API_KEY},HF_DATASETS_CACHE=${HF_DATASETS_CACHE},HF_TOKEN=${HF_TOKEN},ADD_ARG=${ADD_ARG} \
        --command=\"bash ${COMMAND_FILE}\""

    if [ -z "$dep" ]; then
        eval $common_args
    else
        eval $common_args --dependency=afterany:$dep
    fi
    cat $TMP_JOBID_FILE
}

# Submit the first job
submit
last_job_id=$(cat $TMP_JOBID_FILE)
echo "Submitted job $last_job_id"

# Submit the remaining jobs, each dependent on the previous one
for i in $(seq 2 $N_RUNS); do
    submit $last_job_id
    last_job_id=$(cat $TMP_JOBID_FILE)
    echo "Submitted job $last_job_id"
done

rm $TMP_JOBID_FILE
echo "All $N_RUNS jobs submitted successfully."