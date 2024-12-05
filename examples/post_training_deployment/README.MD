

# Zero-Shot Foundation Model Specialization via OSF

## Overview
This repository demonstrates OSF's (Optimized Supernet Formation) capability for zero-shot model specialization. Our approach enables automatic generation of resource-efficient model variants without additional training, making it particularly suitable for edge device deployment.

## Key Contributions
- Architecture-agnostic supernet training methodology
- Efficient zero-shot subnet extraction (>10¹² possible configurations)
- Direct deployment capability without fine-tuning
- Comprehensive evaluation across multiple architectures and tasks

## Prerequisites

### Environment Setup
```bash
conda create -n osf python=3.10
conda activate osf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Pre-trained Supernet Models

Available checkpoints through Hugging Face:

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

## Empirical Validation

### Tutorial
We provide a comprehensive Jupyter notebook demonstrating:
- Supernet initialization and configuration
- Zero-shot subnet extraction methodology
- Performance evaluation protocols
- Resource-accuracy trade-off analysis

**Tutorial:** [vit_zero_shot_specialization_turorial.ipynb](./vit_zero_shot_specialization_turorial.ipynb)
