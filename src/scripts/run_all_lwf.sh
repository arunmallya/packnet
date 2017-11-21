#!/bin/bash
# Script that calls function to perform LwF. 
# Requires 2 GPUs: 1) To compute targets, 2) To train network.

./scripts/run_lwf.sh csf ../checkpoints/imagenet/vgg16.pt 2,3
