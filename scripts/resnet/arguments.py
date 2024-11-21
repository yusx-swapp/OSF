import argparse

def arguments():
    parser = argparse.ArgumentParser(description="Distributed Training with DDP")
    
    # Essential arguments for model and training
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["resnet", "vit"],
        help="Model architecture to use"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ckpts/",
        help="Directory to save the model checkpoints"
    )
    
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet-1k",
        choices=["cifar100", "cifar10", "imagenet-1k"],
        help="Dataset to use for training"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-3,
        help="Learning rate for the optimizer"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for optimizer"
    )
    
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs for learning rate warmup"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=256,
        help="Training batch size per GPU"
    )
    
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=256,
        help="Evaluation batch size per GPU"
    )
    
    # Optimizer and scheduler settings
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd"],
        help="Optimizer to use"
    )
    
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "step"],
        help="Learning rate scheduler"
    )
    
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for scheduler"
    )
    
    # Logging and saving
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Number of steps between logging outputs"
    )
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    
    # Distributed training specific
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Model loading and caching
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        required=True,
        help="Huggingface token for ImageNet access"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="Cache directory for datasets and models"
    )
    
    # Performance optimization
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed-precision training"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers per GPU"
    )
    
    parser.add_argument(
        "--elastic_config",
        type=str,
        default=None,
        help="Path to elastic configuration file"
    )

    parser.add_argument(
        "--val_subnet_samples",
        type=int,
        default=1,
        help="Number of subnets to evaluate per batch"
    )

    args = parser.parse_args()
    return args