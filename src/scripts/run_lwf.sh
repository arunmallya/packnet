#!/bin/bash
# Runs the Learning without Forgetting (LwF) method on given task sequence.
# ./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3

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
LOADNAME=$2
GPU_IDS=$3

for (( i=0; i<${#ORDER}; i++ )); do
  dataset=${DATASETS[${ORDER:$i:1}]}

  mkdir ../checkpoints/$dataset/lwf_$ORDER/
  mkdir ../logs/$dataset/lwf_$ORDER/

  if [ $i -eq 0 ]
  then
    loadname=$LOADNAME
  else
    loadname=$ft_savename'.pt'
  fi

  tag=$ORDER
  ft_savename=../checkpoints/$dataset/lwf_$ORDER/$tag
  logname=../logs/$dataset/lwf_$ORDER/$tag

  ##############################################################################
  # Train on current dataset.
  ##############################################################################
  CUDA_VISIBLE_DEVICES=$GPU_IDS python lwf.py --mode finetune \
    --dataset $dataset --num_outputs ${NUM_OUTPUTS[$dataset]} \
    --loadname $loadname \
    --lr 1e-3 --lr_decay_every 10 --lr_decay_factor 0.1 --finetune_epochs 20 \
    --save_prefix $ft_savename | tee $logname'.txt'
done
