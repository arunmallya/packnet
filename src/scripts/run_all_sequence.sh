#!/bin/bash
# Script that calls function to add tasks in sequence using the iterative 
# pruning + re-training method.
# Usage:
# ./scripts/run_all_sequence.sh 3 vgg16 1
# ./scripts/run_all_sequence.sh 2 vgg16bn 1
# ./scripts/run_all_sequence.sh 1 resnet50 1
# ./scripts/run_all_sequence.sh 0 densenet121 1

GPU_ID=$1
ARCH=$2
RUN_ID=$3
TAG='nobias-nobn'
EXTRA_FLAGS=''
# EXTRA_FLAGS='--train_biases --train_bn'

mkdir ../checkpoints/cubs_cropped
mkdir ../checkpoints/stanford_cars_cropped
mkdir ../checkpoints/flowers
mkdir ../logs/cubs_cropped
mkdir ../logs/stanford_cars_cropped
mkdir ../logs/flowers

./scripts/run_sequence.sh csf 0.75,0.75,-1 \
  ../checkpoints/imagenet/$ARCH'_pruned_0.5_'$RUN_ID'_final.pt' \
  $GPU_ID $ARCH'_0.5-'$TAG'_'$RUN_ID $EXTRA_FLAGS

