import os
import torch
import numpy as np
from datasets import load_dataset
import functools
import evaluate
from transformers import AutoImageProcessor, AutoModelForImageClassification
from arguments import arguments
import torch.multiprocessing as mp
from osf import GraphIR
from osf.utils import calculate_params
import resnet_elastic
import copy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm
# from torch.amp import autocast, GradScaler
import math
import time
from pathlib import Path
import gc
def get_optimizer(model, args):
    """Initialize optimizer"""
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    return optimizer

def get_scheduler(optimizer, args, num_training_steps):
    """Initialize learning rate scheduler with warmup"""
    num_warmup_steps = args.warmup_epochs * num_training_steps // args.epochs
    
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=num_training_steps,
            pct_start=args.warmup_epochs/args.epochs,
            anneal_strategy='cos',
            final_div_factor=args.lr/args.min_lr,
            div_factor=25
        )
    elif args.lr_scheduler == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:  # step
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return scheduler

def setup():
    """Initialize the distributed training environment"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    
    # Set cuda device first
    device = torch.device(f"cuda:{local_rank}")
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    if global_rank == 0:
        print(f"Training with {world_size} GPUs")
    
    return local_rank, world_size, global_rank, device

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, args, local_rank):
    """Training loop for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Get total dataset size for proper logging
    dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * args.per_device_train_batch_size * dist.get_world_size()
    
    if local_rank == 0:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        start_time = time.time()
    else:
        progress_bar = train_loader
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Mixed precision training
        # with autocast(enabled=args.fp16):
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        
        # Scale loss and backward pass
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        
        # Update learning rate
        scheduler.step()
        
        # Compute accuracy
        with torch.no_grad():
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        total_loss += loss.item()
        
        if batch_idx % args.log_interval == 0 and local_rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            processed_samples = batch_idx * args.per_device_train_batch_size * dist.get_world_size()
            speed = args.log_interval * args.per_device_train_batch_size * dist.get_world_size() / (time.time() - start_time)
            print(f'Epoch: {epoch} [{processed_samples}/{dataset_size} '
                  f'({100. * processed_samples / dataset_size:.0f}%)]\t'
                  f'Loss: {loss.item():.4f}\t'
                  f'Acc: {100. * correct / total:.2f}%\t'
                  f'LR: {current_lr:.6f}\t'
                  f'Speed: {speed:.1f} samples/sec')
            start_time = time.time()
    
    metrics = torch.tensor([total_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics.tolist()
    
    avg_loss = total_loss / (len(train_loader) * dist.get_world_size())
    accuracy = 100. * correct / total
    
    
    # Clear batch variables to avoid memory buildup
    del images, labels, outputs, loss, predicted
    gc.collect()
    
    
    return avg_loss, accuracy

def train_epoch_supernet(supernet, train_loader, optimizer, scheduler, criterion, device, epoch, args, local_rank, ir):
    """Training loop for supernet with dynamic subnet sampling."""
    supernet.train()
    supernet = supernet.cpu()  # Keep supernet on CPU
    total_loss = 0
    correct = 0
    total = 0
    
    dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * args.per_device_train_batch_size * dist.get_world_size()
    
    if local_rank == 0:
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        start_time = time.time()
    else:
        progress_bar = train_loader
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # Sample subnet configuration
        sampled_configs = ir.sample_elastic_configs()

        # Create and move subnet to GPU
        subnet = ir.create_subnet(sampled_configs)
        subnet = subnet.to(device)
        if local_rank == 0:
            step_start_time = time.time()
        
        # Forward pass
        outputs = subnet(images)
        loss = criterion(outputs.logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Apply gradients to supernet
        ir.apply_subnet_grads(subnet)
        ir.supernet.to(device)
        # Sync gradients across ranks
        for param in supernet.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= dist.get_world_size()
        
        # Update supernet weights
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        ir.supernet.to('cpu')
        if local_rank == 0:
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            print(f"Step time: {step_time:.4f} seconds")
            
        # Compute accuracy
        with torch.no_grad():
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        total_loss += loss.item()
        
        # Clean up GPU memory
        del subnet
        torch.cuda.empty_cache()
        
        if batch_idx % args.log_interval == 0 and local_rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            processed_samples = batch_idx * args.per_device_train_batch_size * dist.get_world_size()
            speed = args.log_interval * args.per_device_train_batch_size * dist.get_world_size() / (time.time() - start_time)
            print(f'Epoch: {epoch} [{processed_samples}/{dataset_size} '
                  f'({100. * processed_samples / dataset_size:.0f}%)]\t'
                  f'Loss: {loss.item():.4f}\t'
                  f'Acc: {100. * correct / total:.2f}%\t'
                  f'LR: {current_lr:.6f}\t'
                  f'Speed: {speed:.1f} samples/sec')
            start_time = time.time()
    
    # Gather metrics from all processes
    metrics = torch.tensor([total_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    total_loss, correct, total = metrics.tolist()
    
    avg_loss = total_loss / (len(train_loader) * dist.get_world_size())
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy



@torch.no_grad()
def validate_supernet(supernet, val_loader, criterion, device, local_rank, args, ir):
    """Validation loop for supernet using sampled subnets."""
    supernet.eval()
    supernet = supernet.cpu()  # Keep supernet on CPU
    
    # Sample subnets once before validation
    num_samples = args.val_subnet_samples
    all_val_losses = []
    all_corrects = []
    all_totals = []
    
    for sample_idx in range(num_samples):
        val_loss = 0
        correct = 0
        total = 0
        
        # Sample one subnet configuration
        sampled_configs = ir.sample_elastic_configs()
        subnet = ir.create_subnet(sampled_configs)
        subnet = subnet.to(device)
        
        if local_rank == 0:
            val_loader_iter = tqdm(val_loader, desc=f"Validation Subnet {sample_idx+1}/{num_samples}")
        else:
            val_loader_iter = val_loader
        
        for batch in val_loader_iter:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            outputs = subnet(images)
            loss = criterion(outputs.logits, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # if sample_idx == 0:  # Only print params once
        params = calculate_params(subnet)
        print(f"Subnet {sample_idx+1}/{num_samples}, In rank: {local_rank}, params: {params}")
        
        all_val_losses.append(val_loss)
        all_corrects.append(correct)
        all_totals.append(total)
        print(f"Subnet {sample_idx+1}/{num_samples}, In rank: {local_rank}, val loss: {val_loss}, correct: {correct}, total: {total}, accuracy: {100. * correct / total:.2f}%")
        del subnet
        torch.cuda.empty_cache()
    
    # Average results across all sampled subnets
    val_loss = sum(all_val_losses) / num_samples
    correct = sum(all_corrects) / num_samples
    total = all_totals[0]  # All totals should be the same
    
    # Gather metrics from all processes
    metrics = torch.tensor([val_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    val_loss, correct, total = metrics.tolist()
    
    val_loss = val_loss / len(val_loader) / dist.get_world_size()
    accuracy = 100. * correct / total
    
    if local_rank == 0:
        print(f'\nValidation set: Average loss: {val_loss:.4f}, '
              f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    
    return val_loss, accuracy
@torch.no_grad()
def validate(model, val_loader, criterion, device, local_rank, args):
    """Validation loop with mixed precision"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    if local_rank == 0:
        val_loader = tqdm(val_loader, desc="Validation")
    
    for batch in val_loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        
        # with autocast(enabled=args.fp16):
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        
        val_loss += loss.item()
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Gather metrics from all processes
    metrics = torch.tensor([val_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    val_loss, correct, total = metrics.tolist()
    
    val_loss = val_loss / len(val_loader) / dist.get_world_size()
    accuracy = 100. * correct / total
    
    if local_rank == 0:
        print(f'\nValidation set: Average loss: {val_loss:.4f}, '
              f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
            # Clear batch variables to avoid memory buildup
    
    del images, labels, outputs, loss, predicted
    
    
    
    return val_loss, accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, args, is_best=False):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_accuracy': best_accuracy,
    }
    
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save latest checkpoint
    if epoch % args.save_interval == 0:
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
        
    # Save elastic config if used
    if args.elastic_config and is_best:
        ir.model.save_elastic_config(checkpoint_dir / "elastic_space.json")

def main(args):
    local_rank, world_size, global_rank, device = setup()
    
    # Set random seed
    torch.manual_seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    
    if global_rank == 0:
        print("Loading dataset...")
    
    # Load dataset and processor
    model_name = "microsoft/resnet-50"
    processor_name = "microsoft/resnet-50"
    
    if args.huggingface_token:
        from huggingface_hub import login
        login(args.huggingface_token, add_to_git_credential=True)

    dataset = load_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        split=['train', 'validation']
    )
    
    dataset = {
        'train': dataset[0],
        'validation': dataset[1]
    }

    if args.dataset == "imagenet-1k":
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")

    labels = dataset['train'].features["label"].names
    processor = AutoImageProcessor.from_pretrained(processor_name, cache_dir=args.cache_dir)
    
    # Transform function
    def transform(example_batch, processor):
        inputs = processor([x.convert("RGB") for x in example_batch["img"]], return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs
    
    prepared_ds = {
        split: dataset[split].with_transform(functools.partial(transform, processor=processor))
        for split in ['train', 'validation']
    }

    # Create data loaders with DistributedSampler
    train_sampler = DistributedSampler(
        prepared_ds['train'],
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        prepared_ds['validation'],
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        prepared_ds['train'],
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        prepared_ds['validation'],
        batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    # Initialize model
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=True
    )

    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply elastic configurations if specified
    
    ir = GraphIR(model)
    configs = copy.deepcopy(resnet_elastic.ELASTIC_CONFIGS)
    for module_name, config in configs.items():
        ir.set_elastic_config(module_name, config)
    # Only rank 0 samples configuration
    if dist.get_rank() == 0:
        sampled_configs = ir.sample_elastic_configs()
        try:
            torch.save(sampled_configs, os.path.join(args.output_dir, "sampled_configs.pth"))
        except:
            #save the json
            import json
            with open(os.path.join(args.output_dir, "sampled_configs.json"), "w") as f:
                json.dump(sampled_configs, f)
        # sampled_configs = ir.sample_min_elastic_config()
    else:
        sampled_configs = None

    # Broadcast sampled configs from rank 0 to all ranks
    if dist.get_world_size() > 1:
        sampled_configs = [sampled_configs if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(sampled_configs, src=0)
        sampled_configs = sampled_configs[0]

    # Create identical subnet on all ranks
    model = ir.create_subnet(sampled_configs)
    
    try:
        from osf.utils import calculate_params
        print("Params: ", calculate_params(model))
    except:
        pass
    
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # Initialize optimizer, scheduler, and scaler
    optimizer = get_optimizer(model, args)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(optimizer, args, num_training_steps)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        if global_rank == 0:
            print(f"Loading checkpoint from {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        try:
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']

        except:
            print("Failed to load checkpoint")
            

    model = torch.compile(model)    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch, args, local_rank
        )
        
        val_loss, val_acc = validate(model, val_loader, criterion, device, local_rank, args)
        
        # Save checkpoint (only on rank 0)
        if local_rank == 0:
            is_best = val_acc > best_accuracy
            best_accuracy = max(val_acc, best_accuracy)
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, best_accuracy, args, is_best
            )
            
        # Synchronize all processes to prevent memory buildup before next epoch
        dist.barrier()
        gc.collect()
    

    if local_rank == 0:
        print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    dist.destroy_process_group()
def supernet_train_main(args):
    """Main function for supernet training with distributed data parallel."""
    local_rank, world_size, global_rank, device = setup()
    
    # Set random seed
    torch.manual_seed(args.seed + global_rank)
    np.random.seed(args.seed + global_rank)
    
    if global_rank == 0:
        print("Loading dataset...")
    
    # Load dataset and processor
    model_name = "microsoft/resnet-50"
    processor_name = "microsoft/resnet-50"
    
    if args.huggingface_token:
        from huggingface_hub import login
        login(args.huggingface_token, add_to_git_credential=True)


    if args.dataset in ["cifar100", "cifar10"]:
        dataset = load_dataset(
            args.dataset, cache_dir=args.cache_dir, trust_remote_code=True
        )
        
        if args.dataset == "cifar100":
            dataset = dataset.rename_column("fine_label", "label")

        train_val = dataset["train"].train_test_split(
            test_size=0.2, stratify_by_column="label", seed=123
        )
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    elif args.dataset == "imagenet-1k":
        # Load and prepare dataset
        dataset = load_dataset(
            args.dataset,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
            split=['train', 'validation']
        )

    
        dataset = {
            'train': dataset[0],
            'validation': dataset[1]
        }

    
        dataset['train'] = dataset['train'].rename_column("image", "img")
        dataset['validation'] = dataset['validation'].rename_column("image", "img")

    labels = dataset['train'].features["label"].names
    processor = AutoImageProcessor.from_pretrained(processor_name, cache_dir=args.cache_dir)
    
    # Transform function
    def transform(example_batch, processor):
        inputs = processor([x.convert("RGB") for x in example_batch["img"]], return_tensors="pt")
        inputs["labels"] = example_batch["label"]
        return inputs
    
    prepared_ds = {
        split: dataset[split].with_transform(functools.partial(transform, processor=processor))
        for split in ['train', 'validation']
    }

    # Create data loaders with DistributedSampler
    train_sampler = DistributedSampler(
        prepared_ds['train'],
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        prepared_ds['validation'],
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        prepared_ds['train'],
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        prepared_ds['validation'],
        batch_size=args.per_device_eval_batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )

    # Initialize supernet
    supernet = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        cache_dir=args.cache_dir,
        ignore_mismatched_sizes=True
    )
    
    # Initialize Graph IR and set elastic configurations
    ir = GraphIR(supernet)
    configs = copy.deepcopy(resnet_elastic.ELASTIC_CONFIGS_CIFAR10)
    for module_name, config in configs.items():
        ir.set_elastic_config(module_name, config)

    # Keep supernet on CPU to save memory
    supernet = supernet.cpu()

    # Create output directory
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save elastic configurations for reproducibility
    if global_rank == 0:
        torch.save(configs, checkpoint_dir / "elastic_configs.pth")

    # Initialize optimizer and scheduler
    optimizer = get_optimizer(supernet, args)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(optimizer, args, num_training_steps)
    criterion = torch.nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        if global_rank == 0:
            print(f"Loading checkpoint from {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location='cpu')
        try:
            supernet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
        except:
            print("Failed to load checkpoint")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch_supernet(
            supernet, train_loader, optimizer, scheduler,
            criterion, device, epoch, args, local_rank, ir
        )
        
        val_loss, val_acc = validate_supernet(
            supernet, val_loader, criterion, device, 
            local_rank, args, ir
        )

        # Save checkpoint (only on rank 0)
        if local_rank == 0:
            is_best = val_acc > best_accuracy
            best_accuracy = max(val_acc, best_accuracy)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': supernet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
            }
            
            if epoch % args.save_interval == 0:
                torch.save(checkpoint, checkpoint_dir / f'supernet_checkpoint_epoch_{epoch}.pth')
            
            if is_best:
                torch.save(checkpoint, checkpoint_dir / 'best_supernet.pth')
                
        # Synchronize processes before next epoch
        dist.barrier()
        gc.collect()

    if local_rank == 0:
        print(f"Supernet training completed. Best validation accuracy: {best_accuracy:.2f}%")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    args = arguments()
    setattr(args,"training_mode", "supernet")
    if args.training_mode == "supernet":
        supernet_train_main(args)
    else:
        main(args)  # Original single architecture training
