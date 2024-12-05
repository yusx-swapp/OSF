
# CLIP Architecture Optimization via OSF

## Overview
This repository demonstrates the application of Optimized Supernet Formation (OSF) to CLIP (Contrastive Language-Image Pre-training) models. OSF enables efficient architecture search while preserving the model's multi-modal capabilities through contrastive learning.

## Objectives
1. Transform pre-trained CLIP models into weight-sharing supernets using OSF's graph-based intermediate representation
2. Generate and evaluate efficient CLIP subnets achieving >30% parameter reduction while maintaining competitive performance on image classification tasks

## Implementation Details

### Environment Setup
```bash
conda create -n osf python=3.10
conda activate osf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Experimental Validation
We provide a comprehensive Jupyter notebook demonstrating:
- Supernet construction from pre-trained CLIP models
- Subnet extraction methodology
- Performance evaluation protocols
- Empirical results analysis

**Tutorial:** [CLIP_img_classification.ipynb](CLIP_img_classification.ipynb)

### Pre-trained Supernet Models
Trained supernet checkpoints are available through Hugging Face:

Architecture | Dataset | Link
-------------|---------|------
Swin-v2-base | CIFAR-10 | [Link](https://huggingface.co/anonymous-429/osf-swin-base-patch4-window7-cifar10)
Swin-v2-base | CIFAR-100 | [Link](https://huggingface.co/anonymous-429/osf-swinv2-base-patch4-window7-cifar100)
CLIP-base | CIFAR-10 | [Link](https://huggingface.co/anonymous-429/osf-clip-base-patch32-cifar10)
CLIP-base | CIFAR-100 | [Link](https://huggingface.co/anonymous-429/osf-clip-base-patch32-cifar100)
Mamba-1.4B | Lambda | [Link](https://huggingface.co/anonymous-429/osf-mamba-1.4b-lambda-hf)
ViT-Base | ImageNet | [Link](https://huggingface.co/anonymous-429/osf-vit-base-patch16-224-imagenet)
ViT-Base | CIFAR-100 | [Link](https://huggingface.co/anonymous-429/osf-vit-base-patch16-224-cifar100)
ViT-Base | CIFAR-10 | [Link](https://huggingface.co/anonymous-429/osf-vit-base-patch16-224-cifar10)

### Training Protocol

#### Single-GPU Training
```bash
python scripts/train_clip_img_classification.py \
    --model clip \
    --save_dir ckpts/clip-cifar10 \
    --dataset cifar10 \
    --num_shards 16 \
    --lr 2e-5 \
    --batch_size 64 \
    --log_interval 100 \
    --huggingface_token "<token>" \  # Required for ImageNet and model hub uploads
    --elastic_config scripts/clip_elastic_space.json
```

#### Multi-GPU Distributed Training
```bash
torchrun --nproc_per_node=N --nnodes=1 scripts/train_clip_img_classification.py \
    --model clip \
    --save_dir ckpts/clip-cifar10 \
    --dataset cifar10 \
    --num_shards 16 \
    --lr 2e-5 \
    --batch_size 64 \
    --log_interval 100 \
    --huggingface_token "<token>" \
    --elastic_config scripts/clip_elastic_space.json
```

## Citation
This implementation builds upon the CLIP architecture. If you use this code, please cite both our work and the original CLIP paper:

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and 
          Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and 
          Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```
