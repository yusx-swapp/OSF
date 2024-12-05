
# Mamba SSM Optimization via OSF

## Overview
This repository demonstrates the application of Optimized Supernet Formation (OSF) to Mamba Selective State Space Models (SSMs). We showcase efficient architecture search while maintaining model performance through our graph-based intermediate representation approach.

## Objectives
1. Transform Mamba SSM models into weight-sharing supernets using OSF
2. Generate and evaluate efficient Mamba subnets achieving >800M parameter reduction while maintaining competitive performance on the Lambda dataset

## Prerequisites

### Language Model Evaluation Framework
This implementation requires the Language Model Evaluation Harness for performance assessment:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

For comprehensive evaluation protocols, refer to the [official documentation](https://github.com/EleutherAI/lm-evaluation-harness).

### Environment Setup
```bash
conda create -n osf python=3.10
conda activate osf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Implementation Resources

### Tutorial
We provide a detailed Jupyter notebook demonstrating:
- Supernet construction methodology
- Subnet extraction protocols
- Performance evaluation procedures
- Results analysis

**Tutorial:** [mamba_lm_harness.ipynb](mamba_lm_harness.ipynb)

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

## Empirical Validation
For detailed experimental results and performance analysis, refer to our comprehensive tutorial: [mamba_lm_harness.ipynb](mamba_lm_harness.ipynb)

## Citation
This implementation builds upon the Mamba architecture and evaluation framework. If you use this code, please cite the following works:

```bibtex
@misc{gu2023mamba,
      title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces}, 
      author={Albert Gu and Tri Dao},
      year={2023},
      eprint={2312.00752},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and 
                 Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and 
                 Hsu, Jeffrey and Le Noac'h, Alain and others},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```

