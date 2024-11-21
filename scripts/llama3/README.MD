
# Llama 3.2 OSF Experiments

This directory contains scripts and configurations for running OSF (Operator Searching Framework) experiments on Llama 3.2 models. The experiments include LoRA fine-tuning and evaluation using the EleutherAI evaluation harness.

## Setup

1. First, download the Llama 3.2 3B Instruct model:
```bash
tune download meta-llama/Llama-3.2-3B-Instruct --output-dir /tmp/Llama-3.2-3B-Instruct --ignore-patterns "original/consolidated.00.pth"
```

2. Install required dependencies:
```bash
pip install lm_eval>=0.4.2
```

## Running Experiments

### 1. LoRA Fine-tuning with OSF

The `3B_lora.yaml` config is used for multi-device LoRA fine-tuning with OSF. To run on 2 GPUs:

```bash
tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/3B_lora
```

Key configuration options in `3B_lora.yaml`:
- `batch_size`: 4
- `gradient_accumulation_steps`: 4 
- `epochs`: 5
- `lr`: 3e-4
- `lora_rank`: 64
- `lora_alpha`: 128

You can override specific config values through command line:
```bash
tune run --nproc_per_node 2 lora_finetune_distributed --config llama3/3B_lora checkpointer.checkpoint_dir=<YOUR_CHECKPOINT_DIR>
```

### 2. Model Evaluation

The `eleuther_evaluation.yaml` config is used to evaluate models using the EleutherAI evaluation harness. To run evaluation:

```bash
tune run eleuther_eval --config llama3/eleuther_evaluation tasks=["truthfulqa_mc2","commonsense_qa","hellaswag"]
```

Key evaluation tasks:
- `truthfulqa_mc2`
- `commonsense_qa`
- `hellaswag`

You can modify the tasks list or add evaluation parameters:
```bash
tune run eleuther_eval --config llama3/eleuther_evaluation tasks=["hellaswag"] limit=50
```

## File Structure

- `3B_lora.yaml`: Configuration for LoRA fine-tuning
- `eleuther_eval_recipe.py`: Implementation of EleutherAI evaluation recipe
- `eleuther_evaluation.yaml`: Configuration for model evaluation
- `llama3_2_elastic.py`: OSF elastic configurations for Llama 3.2
- `llama3_osf_recipe.py`: Implementation of OSF fine-tuning recipe

