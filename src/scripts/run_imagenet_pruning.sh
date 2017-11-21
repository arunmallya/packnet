#!/bin/bash
# Performs weight-based pruning on ImageNet-trained networks.
# Usage: 
# ./scripts/run_imagenet_pruning.sh 3 vgg16 0.5 1
# ./scripts/run_imagenet_pruning.sh 3 vgg16bn 0.5 1
# ./scripts/run_imagenet_pruning.sh 3 resnet50 0.5 1
# ./scripts/run_imagenet_pruning.sh 3 densenet121 0.5 1

GPU_IDS=$1
ARCH=$2
PRUNE_PERC=$3
RUN_ID=$4

mkdir ../checkpoints/imagenet
mkdir ../logs/imagenet

# Dump the initial model.
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --arch $ARCH --init_dump

# Do the pruning on dumped model.
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py --dataset imagenet \
  --loadname ../checkpoints/imagenet/$ARCH.pt \
  --mode prune --prune_perc_per_layer $PRUNE_PERC --post_prune_epochs 10 \
  --lr 1e-4 --lr_decay_every 5 --lr_decay_factor 0.1 --train_biases --train_bn \
  --save_prefix ../checkpoints/imagenet/$ARCH'_pruned_'$PRUNE_PERC'_'$RUN_ID | tee ../logs/imagenet/$ARCH'_pruned_'$PRUNE_PERC'_'$RUN_ID'.txt'