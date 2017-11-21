## packnet: https://arxiv.org/abs/1711.05769

Pretrained models are available here: https://uofi.box.com/s/zap2p03tnst9dfisad4u0sfupc0y1fxt  
Datasets in PyTorch format are available here: https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq  
The PyTorch-friendly Places365 dataset can be downloaded from http://places2.csail.mit.edu/download.html  
Place models in `checkpoints/` and unzipped datasets in `data/`

|               |   VGG-16 LwF |    VGG-16    |   VGG-16 BN  |   ResNet-50  | DenseNet-121 |
|:-------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| ImageNet      | 36.58 (14.75)| 29.19 (9.90) | 27.10 (8.70) | 24.33 (7.17) | 25.51 (7.85) |
| CUBS          |        34.24 |        22.56 |        20.43 |        19.59 |        20.11 |
| Stanford Cars |        22.07 |        17.09 |        14.92 |        14.03 |        16.18 |
| Flowers       |        12.15 |        11.07 |         8.59 |         8.12 |         9.07 |

Note that the numbers in the [paper](https://arxiv.org/abs/1711.05769) are averaged over multiple runs for each ordering
of datasets. The pretrained models are for a specific dataset addition ordering: (c) CUBS Birds, (s) Stanford Cars, (f) Flowers.

These numbers were obtained by evaluating the models on a Titan X (Pascal).  
Note that numbers on other GPUs might be slightly different (~0.1%) owing to cudnn algorithm selection.  
https://discuss.pytorch.org/t/slightly-different-results-on-k-40-v-s-titan-x/10064

## Training:
Check out the scripts in `src/scripts`.  
Run all code from the `src/` directory, e.g. `./scripts/run_all.sh`

## Eval:
```bash
cd src  # Run everything from src/

# Pruning-based models.
python main.py --mode eval --dataset cubs_cropped \
  --loadname ../checkpoints/csf_0.75,0.75,-1_vgg16_0.5-nobias-nobn_1.pt

# LwF models.
python lwf.py --mode eval --dataset cubs_cropped \
  --loadname ../checkpoints/csf_lwf.pt
```
