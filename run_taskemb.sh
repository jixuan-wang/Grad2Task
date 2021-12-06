#!/bin/bash

source ./path.sh
echo "PROJECT_HOME=${PROJECT_HOME}"
echo "DATASET_HOME=${DATASET_HOME}"
echo "LEOPARD_DATA_DIR=${LEOPARD_DATA_DIR}"

cd ${PROJECT_HOME}

###############################
# Experiment running function #
###############################
run () {
python ${PROJECT_HOME}main_taskemb.py \
    --seed ${SEED_ID} \
    --slurm_job_id ${SLURM_JOB_ID} \
    --model_type ${MODEL_TYPE} \
    --mode ${MODE} \
    --data_dir ${DATA_DIR} \
    --leopard_data_dir ${LEOPARD_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --overwrite_output_dir \
    --cnap_pretrained ${CNAP_PRETRAINED} \
    --bert_pretrained ${BERT_PRETRAINED} \
    --exp_config ${exp_config} \
    --wandb ${WANDB}
}

# export CUDA_VISIBLE_DEVICES=0
export MODEL_TYPE=$1
export MODE=$2
export EXP_ID=$3

export WANDB=None
export SEED_ID=42
export DATA_DIR=${DATASET_HOME}
export CACHE_DIR=${PROJECT_HOME}cache
export CNAP_PRETRAINED=none
export BERT_PRETRAINED=${PROJECT_HOME}output/bert-protonet-euc-bn-05-22-12am/checkpoint-CURRENT-BEST/exp_checkpoint.pt
export exp_config=${PROJECT_HOME}task_emb_exp_config.yaml

if [ ! -z ${EXP_ID} ]; then
    export EXP_ID=-${EXP_ID}
fi

export OUTPUT_DIR=${PROJECT_HOME}output/taskemb-${MODEL_TYPE}${EXP_ID}

echo "Experiment configs:"
cat ${exp_config}
echo 
echo "Start running."
run
