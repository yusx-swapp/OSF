
from typing import Dict
from ofm import ElasticConfig, ElasticityType, ElasticRange, DependencyRule


def get_resnet50_elastic_configs() -> Dict[str, ElasticConfig]:
    """Create elastic configurations for ResNet-50 bottleneck layers.
    Channel ranges: 50% to 100% of original size
    """
    elastic_configs = {}
    
    def create_channel_range(min_val: int, max_val: int) -> ElasticRange:
        """Create channel range with step=8 and divisible by 8 constraint"""
        return ElasticRange(
            min_val=min_val - (min_val % 8),  # Make sure min is divisible by 8
            max_val=max_val,
            step=8,
            constraints=["divisible_by_8"]
        )
    
    def match_channels(x: int) -> int:
        return x

    # Stage 1 (64->256)
    # First block with special handling (64->256)
    elastic_configs["resnet.encoder.stages.0.layers.0"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(64, 64),  # Fixed from embedder
            "out_channels": create_channel_range(128, 256)  # Can be reduced
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.0.layers.0",
                source_param="out_channels",
                target_module="resnet.encoder.stages.0.layers.1",
                target_param="in_channels",
                transform_fn=match_channels
            )
        ],
        init_kwargs={
            "stride": 1,
            "activation": "relu",
            "reduction": 4,
            "downsample_in_bottleneck": False
        }
    )
    
    # Other blocks in stage 1
    for i in range(1, 3):
        elastic_configs[f"resnet.encoder.stages.0.layers.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(128, 256),
                "out_channels": create_channel_range(128, 256)
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.0.layers.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.0.layers.{i+1}" if i < 2 else "resnet.encoder.stages.1.layers.0",
                    target_param="in_channels",
                    transform_fn=match_channels
                )
            ],
            init_kwargs={
                "stride": 1,
                "activation": "relu",
                "reduction": 4,
                "downsample_in_bottleneck": False
            }
        )
    
    # Stage 2 (256->512) with stride=2 in first block
    elastic_configs["resnet.encoder.stages.1.layers.0"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(128, 256),
            "out_channels": create_channel_range(256, 512)
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.1.layers.0",
                source_param="out_channels",
                target_module="resnet.encoder.stages.1.layers.1",
                target_param="in_channels",
                transform_fn=match_channels
            )
        ],
        init_kwargs={
            "stride": 2,  # Spatial reduction
            "activation": "relu",
            "reduction": 4,
            "downsample_in_bottleneck": False
        }
    )
    
    # Other blocks in stage 2
    for i in range(1, 4):
        elastic_configs[f"resnet.encoder.stages.1.layers.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(256, 512),
                "out_channels": create_channel_range(256, 512)
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.1.layers.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.1.layers.{i+1}" if i < 3 else "resnet.encoder.stages.2.layers.0",
                    target_param="in_channels",
                    transform_fn=match_channels
                )
            ],
            init_kwargs={
                "stride": 1,
                "activation": "relu",
                "reduction": 4,
                "downsample_in_bottleneck": False
            }
        )
    
    # Stage 3 (512->1024) with stride=2 in first block
    elastic_configs["resnet.encoder.stages.2.layers.0"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(256, 512),
            "out_channels": create_channel_range(512, 1024)
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.2.layers.0",
                source_param="out_channels",
                target_module="resnet.encoder.stages.2.layers.1",
                target_param="in_channels",
                transform_fn=match_channels
            )
        ],
        init_kwargs={
            "stride": 2,  # Spatial reduction
            "activation": "relu",
            "reduction": 4,
            "downsample_in_bottleneck": False
        }
    )
    
    # Other blocks in stage 3
    for i in range(1, 6):
        elastic_configs[f"resnet.encoder.stages.2.layers.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(512, 1024),
                "out_channels": create_channel_range(512, 1024)
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.2.layers.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.2.layers.{i+1}" if i < 5 else "resnet.encoder.stages.3.layers.0",
                    target_param="in_channels",
                    transform_fn=match_channels
                )
            ],
            init_kwargs={
                "stride": 1,
                "activation": "relu",
                "reduction": 4,
                "downsample_in_bottleneck": False
            }
        )
    
    # Stage 4 (1024->2048) with stride=2 in first block
    elastic_configs["resnet.encoder.stages.3.layers.0"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(512, 1024),
            "out_channels": create_channel_range(1024, 2048)
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.3.layers.0",
                source_param="out_channels",
                target_module="resnet.encoder.stages.3.layers.1",
                target_param="in_channels",
                transform_fn=match_channels
            )
        ],
        init_kwargs={
            "stride": 2,  # Spatial reduction
            "activation": "relu",
            "reduction": 4,
            "downsample_in_bottleneck": False
        }
    )
    
    # Other blocks in stage 4
    for i in range(1, 3):
        elastic_configs[f"resnet.encoder.stages.3.layers.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(1024, 2048),
                "out_channels": create_channel_range(1024, 2048)
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.3.layers.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.3.layers.{i+1}" if i < 2 else "classifier.1",
                    target_param="in_channels" if i < 2 else "in_features",
                    transform_fn=match_channels
                )
            ],
            init_kwargs={
                "stride": 1,
                "activation": "relu",
                "reduction": 4,
                "downsample_in_bottleneck": False
            }
        )
    
        # Stage 4 (1024->2048) last block dependency correction
    elastic_configs[f"resnet.encoder.stages.3.layers.2"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(1024, 2048),
            "out_channels": create_channel_range(1024, 2048)
        },
        dependencies=[
            # Connect to classifier's Linear layer
            DependencyRule(
                source_module="resnet.encoder.stages.3.layers.2",
                source_param="out_channels",
                target_module="classifier.1",  # The Linear layer
                target_param="in_features",    # Linear layer's input dimension
                transform_fn=match_channels
            )
        ],
        init_kwargs={
            "stride": 1,
            "activation": "relu",
            "reduction": 4,
            "downsample_in_bottleneck": False
        }
    )
        # Add config for classifier
    elastic_configs["classifier.1"] = ElasticConfig(
        structural_ranges={
            "in_features": create_channel_range(1024, 2048),  # Match last stage's out_channels
            "out_features": create_channel_range(1000, 1000)  # Fixed number of classes
        },
        init_kwargs={
            "bias": True
        }
    )
    return elastic_configs

ELASTIC_CONFIGS = get_resnet50_elastic_configs()