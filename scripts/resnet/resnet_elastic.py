
from typing import Dict,Optional
from osf import ElasticConfig, ElasticityType, ElasticRange, DependencyRule



def get_resnet50_elastic_configs() -> Dict[str, ElasticConfig]:
    elastic_configs = {}
    
    def create_channel_range(min_val: int, max_val: int) -> ElasticRange:
        if min_val == max_val:
            return ElasticRange(
                min_val=min_val,
                max_val=max_val,
                step=1,
                constraints=[]
            )
        min_val = ((min_val + 7) // 8) * 8
        return ElasticRange(
            min_val=min_val,
            max_val=max_val,
            step=8,
            constraints=["divisible_by_8"]
        )

    # Stage 1 (64->256)
    # First block with shortcut
    elastic_configs["resnet.encoder.stages.0.layers.0.shortcut"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(64, 64),
            "out_channels": create_channel_range(128, 256)
        },
        init_kwargs={"stride": 1},
        dependencies=[]
    )

    # First and second conv layers
    for i in range(2):
        elastic_configs[f"resnet.encoder.stages.0.layers.0.layer.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(64 if i == 0 else 32, 64),
                "out_channels": create_channel_range(32, 64)
            },
            init_kwargs={
                "kernel_size": 1 if i == 0 else 3,
                "stride": 1,
                "activation": "relu"
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.0.layers.0.layer.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.0.layers.0.layer.{i+1}",
                    target_param="in_channels"
                )
            ]
        )

    # Last conv layer (i=2)
    elastic_configs["resnet.encoder.stages.0.layers.0.layer.2"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(32, 64),
            "out_channels": create_channel_range(128, 256)
        },
        init_kwargs={
            "kernel_size": 1,
            "stride": 1,
            "activation": None
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.0.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.0.layers.0.shortcut",
                target_param="out_channels"
            ),
            DependencyRule(
                source_module="resnet.encoder.stages.0.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.0.layers.1.layer.0",
                target_param="in_channels"
            )
        ]
    )

    # Remaining blocks in stage 1 (blocks 1 and 2, no shortcut)
    for block_idx in range(1, 3):
        # First and second conv layers
        for i in range(2):
            elastic_configs[f"resnet.encoder.stages.0.layers.{block_idx}.layer.{i}"] = ElasticConfig(
                structural_ranges={
                    # First conv takes output from previous block, second conv takes reduced channels
                    "in_channels": create_channel_range(256 if i == 0 else 64, 256 if i == 0 else 64),
                    # Both reduce channels for bottleneck
                    "out_channels": create_channel_range(32, 64)
                },
                init_kwargs={
                    "kernel_size": 1 if i == 0 else 3,
                    "stride": 1,
                    "activation": "relu"
                },
                dependencies=[
                    DependencyRule(
                        source_module=f"resnet.encoder.stages.0.layers.{block_idx}.layer.{i}",
                        source_param="out_channels",
                        target_module=f"resnet.encoder.stages.0.layers.{block_idx}.layer.{i+1}",
                        target_param="in_channels"
                    )
                ]
            )
        
        # Last conv layer of each block
        elastic_configs[f"resnet.encoder.stages.0.layers.{block_idx}.layer.2"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(32, 64),
                # Must match output dimension of block 0
                "out_channels": create_channel_range(128, 256)
            },
            init_kwargs={
                "kernel_size": 1,
                "stride": 1,
                "activation": None
            },
            dependencies=[
                # Output must match block 0's output dimension for residual
                DependencyRule(
                    source_module="resnet.encoder.stages.0.layers.0.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.0.layers.{block_idx}.layer.2",
                    target_param="out_channels"
                ),
                # Connect to next block's input
                DependencyRule(
                    source_module=f"resnet.encoder.stages.0.layers.{block_idx}.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.0.layers.{block_idx+1}.layer.0" if block_idx < 2 else "resnet.encoder.stages.1.layers.0.layer.0",
                    target_param="in_channels"
                )
            ]
        )
    # Stage 2 (256->512)
    # First block with shortcut and stride=2
    elastic_configs["resnet.encoder.stages.1.layers.0.shortcut"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(128, 256),
            "out_channels": create_channel_range(256, 512)
        },
        init_kwargs={"stride": 2},
        dependencies=[   # Need to match Stage 1's last block output
            DependencyRule(
                source_module="resnet.encoder.stages.0.layers.2.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.1.layers.0.shortcut",
                target_param="in_channels"
            ),
        ]
    )

    # First and second conv layers
    for i in range(2):
        elastic_configs[f"resnet.encoder.stages.1.layers.0.layer.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(256 if i == 0 else 64, 256 if i == 0 else 128),
                "out_channels": create_channel_range(64, 128)
            },
            init_kwargs={
                "kernel_size": 1 if i == 0 else 3,
                "stride": 2 if i == 1 else 1,
                "activation": "relu"
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.1.layers.0.layer.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.1.layers.0.layer.{i+1}",
                    target_param="in_channels"
                )
            ]
        )

    # Last conv layer
    elastic_configs["resnet.encoder.stages.1.layers.0.layer.2"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(64, 128),
            "out_channels": create_channel_range(256, 512)
        },
        init_kwargs={
            "kernel_size": 1,
            "stride": 1,
            "activation": None
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.1.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.1.layers.0.shortcut",
                target_param="out_channels"
            ),
            DependencyRule(
                source_module="resnet.encoder.stages.1.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.1.layers.1.layer.0",
                target_param="in_channels"
            )
        ]
    )
    # Add after Stage 2's first block setup:

    # Remaining blocks in stage 2 (3 more blocks, no shortcut)
    for block_idx in range(1, 4):
        # First and second conv layers
        for i in range(2):
            elastic_configs[f"resnet.encoder.stages.1.layers.{block_idx}.layer.{i}"] = ElasticConfig(
                structural_ranges={
                    # First conv takes output from previous block, second conv takes reduced channels
                    "in_channels": create_channel_range(512 if i == 0 else 128, 512 if i == 0 else 128),
                    "out_channels": create_channel_range(64, 128)
                },
                init_kwargs={
                    "kernel_size": 1 if i == 0 else 3,
                    "stride": 1,
                    "activation": "relu"
                },
                dependencies=[
                    DependencyRule(
                        source_module=f"resnet.encoder.stages.1.layers.{block_idx}.layer.{i}",
                        source_param="out_channels",
                        target_module=f"resnet.encoder.stages.1.layers.{block_idx}.layer.{i+1}",
                        target_param="in_channels"
                    )
                ]
            )
        
        # Last conv layer of each block
        elastic_configs[f"resnet.encoder.stages.1.layers.{block_idx}.layer.2"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(64, 128),
                "out_channels": create_channel_range(256, 512)  # Must match stage 2's first block output
            },
            init_kwargs={
                "kernel_size": 1,
                "stride": 1,
                "activation": None
            },
            dependencies=[
                # Match stage 2's first block output dimension for residual
                DependencyRule(
                    source_module="resnet.encoder.stages.1.layers.0.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.1.layers.{block_idx}.layer.2",
                    target_param="out_channels"
                ),
                # Connect to next block's input
                DependencyRule(
                    source_module=f"resnet.encoder.stages.1.layers.{block_idx}.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.1.layers.{block_idx+1}.layer.0" if block_idx < 3 else "resnet.encoder.stages.2.layers.0.layer.0",
                    target_param="in_channels"
                )
            ]
        )

    # Stage 3 (512->1024)
    # First block with shortcut and stride=2
    elastic_configs["resnet.encoder.stages.2.layers.0.shortcut"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(256, 512),
            "out_channels": create_channel_range(512, 1024)
        },
        init_kwargs={"stride": 2},
        dependencies=[
            # Add dependency to match previous stage output
            DependencyRule(
                source_module="resnet.encoder.stages.1.layers.3.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.2.layers.0.shortcut",
                target_param="in_channels"
            )
        ]
    )

    # First and second conv layers
    for i in range(2):
        elastic_configs[f"resnet.encoder.stages.2.layers.0.layer.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(512 if i == 0 else 128, 512 if i == 0 else 256),
                "out_channels": create_channel_range(128, 256)
            },
            init_kwargs={
                "kernel_size": 1 if i == 0 else 3,
                "stride": 2 if i == 1 else 1,
                "activation": "relu"
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.2.layers.0.layer.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.2.layers.0.layer.{i+1}",
                    target_param="in_channels"
                )
            ]
        )

    # Last conv layer
    elastic_configs["resnet.encoder.stages.2.layers.0.layer.2"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(128, 256),
            "out_channels": create_channel_range(512, 1024)
        },
        init_kwargs={
            "kernel_size": 1,
            "stride": 1,
            "activation": None
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.2.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.2.layers.0.shortcut",
                target_param="out_channels"
            ),
            DependencyRule(
                source_module="resnet.encoder.stages.2.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.2.layers.1.layer.0",
                target_param="in_channels"
            )
        ]
    )

    # Stage 3 remaining blocks
    for block_idx in range(1, 6):
        # First two conv layers
        for i in range(2):
            elastic_configs[f"resnet.encoder.stages.2.layers.{block_idx}.layer.{i}"] = ElasticConfig(
                structural_ranges={
                    "in_channels": create_channel_range(1024 if i == 0 else 256, 1024 if i == 0 else 256),
                    "out_channels": create_channel_range(256, 256)
                },
                init_kwargs={
                    "kernel_size": 1 if i == 0 else 3,
                    "stride": 1,
                    "activation": "relu"
                },
                dependencies=[
                    DependencyRule(
                        source_module=f"resnet.encoder.stages.2.layers.{block_idx}.layer.{i}",
                        source_param="out_channels",
                        target_module=f"resnet.encoder.stages.2.layers.{block_idx}.layer.{i+1}",
                        target_param="in_channels"
                    )
                ]
            )
        
        # Last conv layer
        elastic_configs[f"resnet.encoder.stages.2.layers.{block_idx}.layer.2"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(256, 256),
                "out_channels": create_channel_range(512, 1024)  # Match stage's first block
            },
            init_kwargs={
                "kernel_size": 1,
                "stride": 1,
                "activation": None
            },
            dependencies=[
                # Match stage 3's first block output
                DependencyRule(
                    source_module="resnet.encoder.stages.2.layers.0.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.2.layers.{block_idx}.layer.2",
                    target_param="out_channels"
                ),
                # Connect to next block
                DependencyRule(
                    source_module=f"resnet.encoder.stages.2.layers.{block_idx}.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.2.layers.{block_idx+1}.layer.0" if block_idx < 5 else "resnet.encoder.stages.3.layers.0.layer.0",
                    target_param="in_channels"
                )
            ]
        )

    # Stage 4 (1024->2048)
    # First block with shortcut and stride=2
    elastic_configs["resnet.encoder.stages.3.layers.0.shortcut"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(512, 1024),
            "out_channels": create_channel_range(1024, 2048)
        },
        init_kwargs={"stride": 2},
        dependencies=[
            # Add dependency to match previous stage output
            DependencyRule(
                source_module="resnet.encoder.stages.2.layers.5.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.3.layers.0.shortcut",
                target_param="in_channels"
            )
        ]
    )

    # First and second conv layers
    for i in range(2):
        elastic_configs[f"resnet.encoder.stages.3.layers.0.layer.{i}"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(1024 if i == 0 else 256, 1024 if i == 0 else 512),
                "out_channels": create_channel_range(256, 512)
            },
            init_kwargs={
                "kernel_size": 1 if i == 0 else 3,
                "stride": 2 if i == 1 else 1,
                "activation": "relu"
            },
            dependencies=[
                DependencyRule(
                    source_module=f"resnet.encoder.stages.3.layers.0.layer.{i}",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.3.layers.0.layer.{i+1}",
                    target_param="in_channels"
                )
            ]
        )

    # Last conv layer
    elastic_configs["resnet.encoder.stages.3.layers.0.layer.2"] = ElasticConfig(
        structural_ranges={
            "in_channels": create_channel_range(256, 512),
            "out_channels": create_channel_range(1024, 2048)
        },
        init_kwargs={
            "kernel_size": 1,
            "stride": 1,
            "activation": None
        },
        dependencies=[
            DependencyRule(
                source_module="resnet.encoder.stages.3.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.3.layers.0.shortcut",
                target_param="out_channels"
            ),
            DependencyRule(
                source_module="resnet.encoder.stages.3.layers.0.layer.2",
                source_param="out_channels",
                target_module="resnet.encoder.stages.3.layers.1.layer.0",
                target_param="in_channels"
            )
        ]
    )


    # Stage 4 remaining blocks
    for block_idx in range(1, 3):
        # First two conv layers
        for i in range(2):
            elastic_configs[f"resnet.encoder.stages.3.layers.{block_idx}.layer.{i}"] = ElasticConfig(
                structural_ranges={
                    "in_channels": create_channel_range(2048 if i == 0 else 512, 2048 if i == 0 else 512),
                    "out_channels": create_channel_range(512, 512)
                },
                init_kwargs={
                    "kernel_size": 1 if i == 0 else 3,
                    "stride": 1,
                    "activation": "relu"
                },
                dependencies=[
                    DependencyRule(
                        source_module=f"resnet.encoder.stages.3.layers.{block_idx}.layer.{i}",
                        source_param="out_channels",
                        target_module=f"resnet.encoder.stages.3.layers.{block_idx}.layer.{i+1}",
                        target_param="in_channels"
                    )
                ]
            )
        
        # Last conv layer
        elastic_configs[f"resnet.encoder.stages.3.layers.{block_idx}.layer.2"] = ElasticConfig(
            structural_ranges={
                "in_channels": create_channel_range(512, 512),
                "out_channels": create_channel_range(1024, 2048)  # Match stage's first block
            },
            init_kwargs={
                "kernel_size": 1,
                "stride": 1,
                "activation": None
            },
            dependencies=[
                # Match stage 4's first block output
                DependencyRule(
                    source_module="resnet.encoder.stages.3.layers.0.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.3.layers.{block_idx}.layer.2",
                    target_param="out_channels"
                ),
                # Connect to next block or classifier
                DependencyRule(
                    source_module=f"resnet.encoder.stages.3.layers.{block_idx}.layer.2",
                    source_param="out_channels",
                    target_module=f"resnet.encoder.stages.3.layers.{block_idx+1}.layer.0" if block_idx < 2 else "classifier.1",
                    target_param="in_channels" if block_idx < 2 else "in_features"
                )
            ]
        )

    # Classifier
    elastic_configs["classifier.1"] = ElasticConfig(
        structural_ranges={
            "in_features": create_channel_range(1024, 2048),
            # "out_features": create_channel_range(10, 10)
        },
        init_kwargs={"bias": True},
        dependencies=[]
    )

    return elastic_configs

ELASTIC_CONFIGS = get_resnet50_elastic_configs()
ELASTIC_CONFIGS_CIFAR10 = get_resnet50_elastic_configs()