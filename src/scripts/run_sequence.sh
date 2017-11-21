#!/bin/bash
# Adds tasks in sequence using the iterative pruning + re-training method.
# Usage:
# ./scripts/run_sequence.sh ORDER PRUNE_STR LOADNAME GPU_IDS RUN_TAG EXTRA_FLAGS
# ./scripts/run_sequence.sh csf 0.75,0.75,-1 ../checkpoints/imagenet/imagenet_pruned_0.5_final.pt 3 nobias_1 

# This is hard-coded to prevent silly mistakes.
declare -A DATASETS
DATASETS["i"]="imagenet"
DATASETS["p"]="places"
DATASETS["s"]="stanford_cars_cropped"
DATASETS["c"]="cubs_cropped"
DATASETS["f"]="flowers"
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imagenet"]="1000"
NUM_OUTPUTS["places"]="365"
NUM_OUTPUTS["stanford_cars_cropped"]="196"
NUM_OUTPUTS["cubs_cropped"]="200"
NUM_OUTPUTS["flowers"]="102"

ORDER=$1
PRUNE_STR=$2
LOADNAME=$3
GPU_IDS=$4
RUN_TAG=$5
EXTRA_FLAGS=$6

IFS=',' read -r -a PRUNE <<< $PRUNE_STR
for (( i=0; i<${#ORDER}; i++ )); do
  dataset=${DATASETS[${ORDER:$i:1}]}
  prune=${PRUNE[$i]}

  mkdir ../checkpoints/$dataset/$ORDER/
  mkdir ../logs/$dataset/$ORDER/

  # Get model to add dataset to.
  if [ $i -eq 0 ]
  then
    loadname=$LOADNAME
  else
    loadname=$prev_pruned_savename'_final.pt'
    if [ ! -f $loadname ]; then
        echo 'Final file not found! Using postprune'
        loadname=$prev_pruned_savename'_postprune.pt'
    fi
  fi

  # Prepare tags and savenames.
  tag=$ORDER'_'$PRUNE_STR
  ft_savename=../checkpoints/$dataset/$ORDER/$tag'_'$RUN_TAG
  pruned_savename=../checkpoints/$dataset/$ORDER/$tag'_'$RUN_TAG'_pruned'
  logname=../logs/$dataset/$ORDER/$tag'_'$RUN_TAG

  ##############################################################################
  # Train on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode finetune $EXTRA_FLAGS \
    --dataset $dataset --num_outputs ${NUM_OUTPUTS[$dataset]} \
    --loadname $loadname \
    --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 20 \
    --save_prefix $ft_savename | tee $logname'.txt'
  
  ##############################################################################
  # Prune on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --mode prune $EXTRA_FLAGS \
    --dataset $dataset --loadname $ft_savename'.pt' \
    --prune_perc_per_layer $prune --post_prune_epochs 10 \
    --lr 1e-4 --lr_decay_every 10 --lr_decay_factor 0.1 \
    --save_prefix $pruned_savename | tee $logname'_pruned.txt'

  prev_pruned_savename=$pruned_savename
done
