# OSF: Architecture-Agnostic Neural Architecture Search with Graph Intermediate Representation

This is the official implementation for the paper:

_Weight-Sharing NAS with Architecture-Agnostic Intermediate Representation_

## Overview
OSF (Optimized Supernet Formation) is a framework for architecture-agnostic neural architecture search that:
- Transforms different types of neural architectures into searchable supernets using graph intermediate representation (IR)
- Enables efficient parallel training of weight-sharing supernets
- Provides automated subnet extraction without architecture-specific rules
- Supports a wide range of architectures including CNNs, Transformers, and State Space Models

## Updates

- [x] Released checkpoints for Mamba, SAM, Swin, and CLIP models
- [x] Released ViT-base supernet checkpoints on HuggingFace model hub
- [x] Added tutorial for converting pre-trained models to supernets
- [x] Added support for Segment Anything Model (SAM)

## Installation

Create a conda environment and install dependencies:

```bash
conda create -n osf python=3.10
conda activate osf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install OSF package:

```bash
cd OSF/
pip install .
```

## Pretrained Supernet Checkpoints


We provide pretrained supernet checkpoints for various architectures. These can be accessed through our HuggingFace model hub:

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

**_You don't need to download the ckpt files, you can use Huggingface Model Card to load the ckpts files directly.
We will show you how to do that in the following section._**

### Quick Start

**We provide detailed instructions and hands-on tutorial for you to validate our zero-shot downsized models:**

- [Example on quickly evaluate ViT supernet with high-performance subnets](./examples/post_training_deployment/vit_zero_shot_specialization_turorial.ipynb)

Multiple examples are provided, including ViT, Swin, CLIP, Mamba, and more. You can explore these examples in the [examples](./examples/) folder.

Besides, we also provide a high-level API for you to quickly generate sunets for your supernet with **2 lines of codes**, as shown in the following example:

```python
from transformers import AutoModelForImageClassification
from ofm import OFM

# Generate downsized models
ckpt_path = "ckpts_repo_name" # Copy the huggingface model hub repo name from above link

model = AutoModelForImageClassification.from_pretrained(
    ckpt_path,
    num_labels=10,
    ignore_mismatched_sizes=True,
)

supernet = OFM(model.to("cpu"))
print("Original FM number of parameters:",supernet.total_params)

#Randomly sample a downsized FM
ds_model, params, config = supernet.random_resource_aware_model()
print("subnetwork params",params)
```

## Training Your Own Supernet

### Single GPU Training

<!-- OFM with its mini-shard training strategy can convert a pre-trained model to a supernet in a fast and efficient way. For instance, you can train a super-ViT on CIFAR-100 with the following command: -->

```bash
python scripts/train_img_classification.py \
    --model vit \
    --dataset cifar100 \
    --num_shards 30 \
    --lr 1e-5 \
    --batch_size 224 \
    --elastic_config configs/elastic_space.json \
    --save_dir checkpoints/cifar100
```

To check the results, you can:

- Check the output information from the terminal console
- Use tensorboard: `tensorboard --logdir log/vit`

### Training on ImageNet

Before you start, you have to be granted access to the ImageNet dataset. You can request and download the dataset from [here](https://huggingface.co/datasets/imagenet-1k).

Set the arguments ` --huggingface_token` to your huggingface token, which should have been granted access to the ImageNet dataset.

```bash
python3 scripts/train_img_classification.py --model vit \
--save_dir 'your_dir'  \
--dataset imagenet-1k \
--num_shards 500 \
--lr 2e-5 \
--batch_size 152 \
--log_interval 500 \
--huggingface_token "your-token-here" \
--elastic_config scripts/elastic_space.json
```

### Distributed Training (Multi-GPU Training)

If you have multiple GPUs, you can use the following command to train the super-FM with distributed training:

```bash
torchrun --nproc_per_node=8 scripts/dist_train.py \
    --model vit \
    --dataset imagenet-1k \
    --num_shards 500 \
    --lr 2e-5 \
    --batch_size 152 \
    --elastic_config configs/elastic_space.json
```

**[Note]**: More APIs and scripts will be posted, please check the [**Updates**](#updates).

## Contact

anonymous

<!-- ## TODO

- [x] ViT pre-trained ckpts
- [x] ViT FL simulation scripts
- [x] Tensorboard logger
- [x] Elastic space APIs for system-heteo
- [x] Load ckpt high-level APIs
- [x] Simulation scripts on GLUE
- [x] ViT CIFAR-100 ckpts
- [x] High level API for real edge-FL
- [x] API for segment anything (SAM)
- [x] Evaluate Scripts for resource-aware models
- [ ] BERT-large, FLAN-T5 ckpts
- [ ] Simulation scripts on SQUAD
- [ ] ONNX and TensorRT APIs for edge
- [ ] Tiny fedlib -->

## Citation

If you find our work is helpful, please kindly support our efforts by citing our paper:

```

under review

```

## Acknowledgement

The experiments of this work is sponsored by **[anonymous institution]** and **[anonymous institution]**.

```

```
