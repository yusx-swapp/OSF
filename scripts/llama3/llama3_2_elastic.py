from typing import Dict
from dataclasses import dataclass
import torch.nn as nn
from typing import Optional, List
from osf import ElasticConfig, ElasticRange, DependencyRule


def get_llama_mlp_elastic_configs(model_path: str = "model.layers") -> Dict[str, ElasticConfig]:
    """Create elastic configurations for Llama MLP layers with proper dependencies."""
    elastic_configs = {}
    
    def create_dim_range(min_val: int, max_val: int) -> ElasticRange:
        """Create dimension range with step=128 and divisible by 128 constraint"""
        return ElasticRange(
            min_val=min_val - (min_val % 128),  # Make sure min is divisible by 128
            max_val=max_val,
            step=128,
            constraints=["divisible_by_128"]
        )
    
    def match_dims(x: int) -> int:
        return x

    # For each MLP layer
    for layer_idx in range(16):
        base_path = f"{model_path}.{layer_idx}.mlp"
        
        # Configure gate_proj
        elastic_configs[f"{base_path}.gate_proj"] = ElasticConfig(
            structural_ranges={
                "in_features": create_dim_range(2048, 2048),  # Fixed input
                "out_features": create_dim_range(2048, 8192)  # Elastic intermediate
            },
            dependencies=[
                # Must match with up_proj output for element-wise multiplication
                DependencyRule(
                    source_module=f"{base_path}.gate_proj",
                    source_param="out_features",
                    target_module=f"{base_path}.up_proj",
                    target_param="out_features",
                    transform_fn=match_dims
                ),
                # Must match with down_proj input
                DependencyRule(
                    source_module=f"{base_path}.gate_proj",
                    source_param="out_features",
                    target_module=f"{base_path}.down_proj",
                    target_param="in_features",
                    transform_fn=match_dims
                )
            ],
            init_kwargs={"bias": False}
        )
        
        # Configure up_proj - will inherit size from gate_proj through dependency
        elastic_configs[f"{base_path}.up_proj"] = ElasticConfig(
            structural_ranges={
                "in_features": create_dim_range(2048, 2048),  # Fixed input
                "out_features": create_dim_range(2048, 8192)  # Will be set by gate_proj
            },
            init_kwargs={"bias": False}
        )
        
        # Configure down_proj - will inherit input size from gate_proj
        elastic_configs[f"{base_path}.down_proj"] = ElasticConfig(
            structural_ranges={
                "in_features": create_dim_range(2048, 8192),  # Will be set by gate_proj
                "out_features": create_dim_range(2048, 2048)  # Fixed output
            },
            init_kwargs={"bias": False}
        )

    return elastic_configs
def get_llama_lora_mlp_elastic_configs(model_path: str = "layers") -> Dict[str, ElasticConfig]:
    """Create elastic configurations for LoRA Llama MLP layers with fixed LoRA parameters.
    
    Fixed LoRA parameters:
    - rank: 64
    - alpha: 128
    - dropout: 0.0
    
    The FeedForward (MLP) has three LoRALinear layers:
    - w1 (gate_proj): 3072 -> 8192
    - w2 (down_proj): 8192 -> 3072
    - w3 (up_proj): 3072 -> 8192
    """
    elastic_configs = {}
    
    def create_dim_range(min_val: int, max_val: int) -> ElasticRange:
        """Create dimension range with step=128 and divisible by 128 constraint"""
        return ElasticRange(
            min_val=min_val - (min_val % 1024),  # Make sure min is divisible by 128
            max_val=max_val,
            step=1024,
            constraints=["divisible_by_128"]
        )
    
    def match_dims(x: int) -> int:
        return x

    # For each layer
    for layer_idx in range(28):  # 28 layers in this version
        base_path = f"{model_path}.{layer_idx}.mlp"
        
        # Configure w1 (gate_proj)
        elastic_configs[f"{base_path}.w1"] = ElasticConfig(
            structural_ranges={
                "in_dim": create_dim_range(3072, 3072),   # Fixed input
                "out_dim": create_dim_range(6144, 8192),  # Elastic intermediate
            },
            dependencies=[
                # Must match with w3 output for element-wise multiplication
                DependencyRule(
                    source_module=f"{base_path}.w1",
                    source_param="out_dim",
                    target_module=f"{base_path}.w3",
                    target_param="out_dim",
                    transform_fn=match_dims
                ),
                # Must match with w2 input
                DependencyRule(
                    source_module=f"{base_path}.w1",
                    source_param="out_dim",
                    target_module=f"{base_path}.w2",
                    target_param="in_dim",
                    transform_fn=match_dims
                )
            ],
            init_kwargs={
                "rank": 64,
                "alpha": 128,
                "dropout": 0.0,
                "use_bias": False,
                "quantize_base": False
            }
        )
        
        # Configure w3 (up_proj)
        elastic_configs[f"{base_path}.w3"] = ElasticConfig(
            structural_ranges={
                "in_dim": create_dim_range(3072, 3072),   # Fixed input
                "out_dim": create_dim_range(6144, 8192),  # Will be set by w1
            },
            init_kwargs={
                "rank": 64,
                "alpha": 128,
                "dropout": 0.0,
                "use_bias": False,
                "quantize_base": False
            }
        )
        
        # Configure w2 (down_proj)
        elastic_configs[f"{base_path}.w2"] = ElasticConfig(
            structural_ranges={
                "in_dim": create_dim_range(6144, 8192),   # Will be set by w1
                "out_dim": create_dim_range(3072, 3072),  # Fixed output
            },
            init_kwargs={
                "rank": 64,
                "alpha": 128,
                "dropout": 0.0,
                "use_bias": False,
                "quantize_base": False
            }
        )

    return elastic_configs

def get_llama_torchtune_mlp_elastic_configs(model_path: str = "layers") -> Dict[str, ElasticConfig]:
    """Create elastic configurations for LoRA Llama MLP layers with fixed LoRA parameters.
    
    
    
    The FeedForward (MLP) has three LoRALinear layers:
    - w1 (gate_proj): 3072 -> 8192
    - w2 (down_proj): 8192 -> 3072
    - w3 (up_proj): 3072 -> 8192
    """
    elastic_configs = {}
    
    def create_dim_range(min_val: int, max_val: int) -> ElasticRange:
        """Create dimension range with step=128 and divisible by 128 constraint"""
        return ElasticRange(
            min_val=min_val - (min_val % 128),  # Make sure min is divisible by 128
            max_val=max_val,
            step=128,
            constraints=["divisible_by_128"]
        )
    
    def match_dims(x: int) -> int:
        return x

    # For each layer
    for layer_idx in range(28):  # 28 layers in this version
        base_path = f"{model_path}.{layer_idx}.mlp"
        
        # Configure w1 (gate_proj)
        elastic_configs[f"{base_path}.w1"] = ElasticConfig(
            structural_ranges={
                "in_features": create_dim_range(3072, 3072),   # Fixed input
                "out_features": create_dim_range(3072, 8192),  # Elastic intermediate
            },
            dependencies=[
                # Must match with w3 output for element-wise multiplication
                DependencyRule(
                    source_module=f"{base_path}.w1",
                    source_param="out_features",
                    target_module=f"{base_path}.w3",
                    target_param="out_features",
                    transform_fn=match_dims
                ),
                # Must match with w2 input
                DependencyRule(
                    source_module=f"{base_path}.w1",
                    source_param="out_features",
                    target_module=f"{base_path}.w2",
                    target_param="in_features",
                    transform_fn=match_dims
                )
            ],
            init_kwargs={
                "bias": False
            }
        )
        
        # Configure w3 (up_proj)
        elastic_configs[f"{base_path}.w3"] = ElasticConfig(
            structural_ranges={
                "in_features": create_dim_range(3072, 3072),   # Fixed input
                "out_features": create_dim_range(3072, 8192),  # Will be set by w1
            },
            init_kwargs={
                "bias": False
            }
        )
        
        # Configure w2 (down_proj)
        elastic_configs[f"{base_path}.w2"] = ElasticConfig(
            structural_ranges={
                "in_features": create_dim_range(3072, 8192),   # Will be set by w1
                "out_features": create_dim_range(3072, 3072),  # Fixed output
            },
            init_kwargs={
                "bias": False
            }
        )

    return elastic_configs

# Module-level variable for easy access
LLAMA_LORA_ELASTIC_CONFIGS = get_llama_lora_mlp_elastic_configs()
LLAMA_ELASTIC_CONFIGS = get_llama_mlp_elastic_configs()
LLAMA_TORCHTUNE_ELASTIC_CONFIGS = get_llama_torchtune_mlp_elastic_configs()
