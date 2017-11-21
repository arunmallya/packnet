#!/bin/bash
# Runs the classifier / individual network baselines.
# Usage:
#  ./scripts/run_baseline_finetuning.sh 3 flowers classifier 1e-3 20 20 1 resnet50
#  ./scripts/run_baseline_finetuning.sh 3 flowers all 1e-3 10 20 1 resnet50 --train_bn

GPU_IDS=$1
DATASET=$2
LAYERS=$3
LR=$4
LR_DECAY_EVERY=$5
NUM_EPOCHS=$6
NUM_RUNS=$7
MODEL=$8
BN_TRAIN=$9

# This is hard-coded to prevent silly mistakes.
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imagenet"]="1000"
NUM_OUTPUTS["places"]="365"
NUM_OUTPUTS["stanford_cars_cropped"]="196"
NUM_OUTPUTS["cubs_cropped"]="200"
NUM_OUTPUTS["flowers"]="102"

mkdir ../checkpoints/$DATASET
mkdir ../logs/$DATASET

for idx in `seq 1 $NUM_RUNS`;
do
    CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
    --mode finetune --finetune_layers $LAYERS \
    --dataset $DATASET --num_outputs ${NUM_OUTPUTS[$DATASET]} \
    --loadname ../checkpoints/imagenet/$MODEL.pt --disable_pruning_mask --train_biases $BN_TRAIN \
    --lr $LR --lr_decay_every $LR_DECAY_EVERY --finetune_epochs $NUM_EPOCHS --lr_decay_factor 0.1 \
    --save_prefix ../checkpoints/$DATASET/$MODEL'_'$idx'_'$LAYERS'_'$LR'_'$LR_DECAY_EVERY'_'$NUM_EPOCHS | tee ../logs/$DATASET/$MODEL'_'$idx'_'$LAYERS'_'$LR'_'$LR_DECAY_EVERY'_'$NUM_EPOCHS.txt
done