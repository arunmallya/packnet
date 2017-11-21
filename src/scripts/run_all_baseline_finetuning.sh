#!/bin/bash
# Runs all baselines: Classifier Only and Individual Network.

declare -a DATASETS=('stanford_cars_cropped' 'cubs_cropped' 'flowers')
declare -a MODELS=('vgg16' 'vgg16bn'  'densenet121' 'resnet50')

GPU_ID=0

# Classifier Only, constant lr works best.
for dataset in "${DATASETS[@]}"
do
  for model in "${MODELS[@]}"
  do
    ./scripts/run_baseline_finetuning.sh $GPU_ID $dataset classifier 1e-3 20 20 1 $model
  done
done

# Individual Network, lr decay works best.
for dataset in "${DATASETS[@]}"
do
  for model in "${MODELS[@]}"
  do
    ./scripts/run_baseline_finetuning.sh $GPU_ID $dataset all 1e-3 10 20 1 $model --train_biases
  done
done
