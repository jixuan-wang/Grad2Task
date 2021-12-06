#!/bin/bash
source ./path.sh
echo "PROJECT_HOME=${PROJECT_HOME}"
echo "DATASET_HOME=${DATASET_HOME}"
echo "LEOPARD_DATA_DIR=${LEOPARD_DATA_DIR}"

cd ${PROJECT_HOME}

# export CUDA_VISIBLE_DEVICES=0
export MODEL_TYPE=$1
export MODE=$2
export EXP_ID=$3
echo $1 $2 $3

###############################
# Experiment running function #
###############################
run () {
python ${PROJECT_HOME}main_baseline.py \
    --slurm_job_id ${SLURM_JOB_ID} \
    --exp_id eid${EXP_ID} \
    --model_type ${MODEL_TYPE} \
    --mode ${MODE} \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --leopard_data_dir ${LEOPARD_DATA_DIR} \
    --max_seq_length 128 \
    --num_shots_support=${NUM_SHOTS_SUPPORT} \
    --num_shots_query=${NUM_SHOTS_QUERY} \
    --num_support_batches ${NUM_SUPPORT_BATCHES} \
    --num_episodes_per_device ${NUM_EPISODES_PER_DEVICE} \
    --num_training_iterations ${NUM_TRAINING_ITERATIONS} \
    --num_training_epochs ${NUM_TRAINING_EPOCHS} \
    --num_iterations_per_optimize_step ${NUM_ITERATIONS_PER_OPTIMIZE_STEP} \
    --bert_linear_size ${BERT_LINEAR_SIZE} \
    --protonet_dist_metric ${PROTONET_DIST_METRIC} \
    --bn_context_size ${CONTEXT_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps_ratio ${WARMUP_RATIO} \
    --weight_decay ${WEIGHT_DECAY} \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID} \
    --adapter_type ${ADAPTER_TYPE} \
    --overwrite_output_dir \
    --val_freq ${VAL_FREQ} \
    --checkpoint_freq ${CHECKPOINT_FREQ} \
    --cnap_freeze_base_model ${CNAP_FREEZE_BASE_MODEL} \
    --cnap_adapt ${CNAP_ADAPT} \
    --cnap_pretrained ${CNAP_PRETRAINED} \
    --early_stop_by ${EARLY_STOP_BY} \
    --early_stop_patience ${EARLY_STOP_PATIENCE} \
    --task_emb_size ${TASK_EMB_SIZE} \
    --fine_tune_epochs ${FINE_TUNE_EPOCHS} \
    --test_per_epoch ${TEST_PER_EPOCH} \
    --checkpoint_path ${checkpoint_path} \
    --wandb ${WANDB}
}


#####################
# Default arguments #
#####################
export SEED_ID=42
export MODEL_NAME_OR_PATH=bert-base-uncased
export DATA_DIR=${DATASET_HOME}
export NUM_SHOTS_SUPPORT=6
export NUM_SHOTS_QUERY=6
export NUM_SUPPORT_BATCHES=1
export NUM_EPISODES_PER_DEVICE=1
export NUM_ITERATIONS_PER_OPTIMIZE_STEP=2
export LEARNING_RATE=5e-5
export NUM_TRAINING_ITERATIONS=1000000
export NUM_TRAINING_EPOCHS=5
export VAL_FREQ=1000
export CHECKPOINT_FREQ=1000
export WEIGHT_DECAY=0
export WARMUP_RATIO=0.0
export CONTEXT_SIZE=768
export BERT_LINEAR_SIZE=256
export CACHE_DIR=${PROJECT_HOME}cache
export CNAP_FREEZE_BASE_MODEL=true
export CNAP_ADAPT=true
export CNAP_PRETRAINED=none
export EARLY_STOP_BY=avg
export EARLY_STOP_PATIENCE=10000000
export TASK_EMB_SIZE=100
export FINE_TUNE_EPOCHS=10
export TEST_PER_EPOCH=false
export WANDB=FineTuneBaseline


###############################
# Overwrite default arguments #
###############################
if [ $1 == "fine-tune-bert" ]; then
    export LEARNING_RATE=5e-5
    export FINE_TUNE_EPOCHS=10
    export checkpoint_path=None
elif [ $1 == "fine-tune-bert-bn" ]; then
    export LEARNING_RATE=5e-3
    export FINE_TUNE_EPOCHS=10
    export checkpoint_path=None
elif [ $1 == "fine-tune-protonet" ]; then
    export LEARNING_RATE=5e-5
    export FINE_TUNE_EPOCHS=10
    export checkpoint_path=output/meta-train-bert-protonet-euc-05-22-3pm/checkpoint-CURRENT-BEST
elif [ $1 == "fine-tune-protonet-bn" ]; then
    export LEARNING_RATE=5e-3
    export FINE_TUNE_EPOCHS=10
    export checkpoint_path=output/meta-train-bert-protonet-euc-bn-05-22-12am/checkpoint-CURRENT-BEST
elif [ $1 == "fine-tune-protonet-bn-film" ]; then
    export LEARNING_RATE=5e-3
    export FINE_TUNE_EPOCHS=10
    export checkpoint_path=output/meta-train-bert-protonet-euc-bn-05-22-12am/checkpoint-CURRENT-BEST
else
    echo "Model not found: $1"
    exit 1
fi

if [ ! -z ${EXP_ID} ]; then
    export EXP_ID=-${EXP_ID}
fi

export OUTPUT_DIR=${PROJECT_HOME}output/${MODEL_TYPE}${EXP_ID}

if [[ ${EXP_ID} == *'debug' ]]; then
    export WANDB=None
    echo "WARNING: wandb is turned off"
fi

run
