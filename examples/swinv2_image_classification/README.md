
# Optimize Swin Transformer via OSF 

## Objectives
This experiment demonstrates:

- [x] OSF-based training methodology for Swin Transformer architectures (Swin and Swinv2)
- [x] Zero-shot subnet extraction achieving >30% parameter reduction while maintaining competitive performance on image classification tasks

## Implementation Resources

### Tutorial
We provide a comprehensive Jupyter notebook demonstrating our methodology and results: 
**[Swin Example](swin_img_classification.ipynb)**

### Pre-trained Supernet Models

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

## Experimental Setup

### Environment Configuration
Refer to the detailed [installation](../../README.md) guide.

```bash
conda create -n osf python=3.10
conda activate osf
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Training Protocols

#### Single-GPU Training
```bash
cd OSF/
python scripts/train_img_classification.py \
--model swinv2 \
--save_dir ckpts/swinv2-cifar10-single \
--dataset cifar10 \  # Options: cifar100, ImageNet
--num_shards 16 \
--lr 2e-5 \
--batch_size 64 \
--log_interval 100 \
--huggingface_token "<token>"  \  # Required for ImageNet and model hub uploads
--elastic_config scripts/swin_elastic_space.json 
```

#### Multi-GPU Distributed Training
```bash
torchrun --nproc_per_node=N --nnodes=1 scripts/train_img_classification.py \
--model swinv2 \
--save_dir ckpts/swinv2-cifar10-multi \
--dataset cifar10 \  # Options: cifar100, ImageNet
--num_shards 16 \
--lr 2e-5 \
--batch_size 64 \
--log_interval 100 \
--huggingface_token "<token>"  \  # Required for ImageNet and model hub uploads
--elastic_config scripts/swin_elastic_space.json
```

## Citation
This implementation builds upon Swin Transformer architectures. Please cite:

```bibtex
@inproceedings{liu2022video,
  title={Video swin transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={3202--3211},
  year={2022}
}

@inproceedings{liu2022swin,
  title={Swin transformer v2: Scaling up capacity and resolution},
  author={Liu, Ze and Hu, Han and Lin, Yutong and Yao, Zhuliang and Xie, Zhenda and Wei, Yixuan and 
          Ning, Jia and Cao, Yue and Zhang, Zheng and Dong, Li and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12009--12019},
  year={2022}
}
```