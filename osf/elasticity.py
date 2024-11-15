from dataclasses import dataclass
from typing import Dict, Any, Union, Tuple, List, Optional, Callable
import numpy as np
from enum import Enum, auto

# Define elasticity types
class ElasticityType(Enum):
    STRUCTURAL = auto()  # Parameter-level elasticity (e.g., channel numbers)
    MODULAR = auto()     # Module-level elasticity (can be removed)


@dataclass
class ElasticRange:
    """Define range and constraints for elastic parameters."""
    min_val: Union[int, float]
    max_val: Union[int, float]
    step: Union[int, float] = 1  # Step size for sampling
    constraints: Optional[List[str]] = None  # e.g., ["divisible_by_8"]
    
    def is_valid(self, value: Union[int, float]) -> bool:
        """Check if a value satisfies the range and constraints."""
        if not (self.min_val <= value <= self.max_val):
            return False
            
        if self.constraints:
            for constraint in self.constraints:
                if constraint == "divisible_by_8" and value % 8 != 0:
                    return False
                # Add more constraints as needed
                
        return True
    def __str__(self):
        constraints_str = f", constraints={self.constraints}" if self.constraints else ""
        return f"[{self.min_val}~{self.max_val}, step={self.step}{constraints_str}]"
    
    def sample(self) -> Union[int, float]:
        """Sample a valid value from the range."""
        if isinstance(self.min_val, int) and isinstance(self.max_val, int):
            possible_values = np.arange(self.min_val, self.max_val + 1, self.step)
            if self.constraints:
                possible_values = [v for v in possible_values if self.is_valid(v)]
            return int(np.random.choice(possible_values))
        else:
            value = np.random.uniform(self.min_val, self.max_val)
            return round(value / self.step) * self.step

# @dataclass
# class ModularConfig:
#     """Configuration for modular elasticity."""
#     removable: bool = False  # Whether module can be removed
#     min_depth: int = 1      # Minimum number of modules to keep
#     max_depth: int = None   # Maximum number of modules (None means no limit)
#     grouping: Optional[str] = None  # Group identifier for related modules
    
#     def __str__(self):
#         depth_str = f"{self.min_depth}~{self.max_depth if self.max_depth else '∞'}"
#         return f"removable={self.removable}, depth={depth_str}, group={self.grouping}"

# @dataclass
# class DependencyRule:
#     """Define how parameters depend on each other."""
#     source_module: str  # Source module name
#     source_param: str   # Source parameter name
#     target_module: str  # Target module name
#     target_param: str   # Target parameter name
#     transform_fn: Optional[Callable] = None  # Optional transformation function
#     reroute_rules: Optional[Dict[str, str]] = None  # Maps source outputs to target inputs when module is removed

    
#     def apply(self, source_value: Any) -> Any:
#         """Apply transformation to source value."""
#         if self.transform_fn:
#             return self.transform_fn(source_value)
#         return source_value
    
#     def __str__(self):
#         transform = "transform" if self.transform_fn else "direct"
#         return f"{self.source_module}.{self.source_param} → {self.target_module}.{self.target_param} ({transform})"
@dataclass
class DependencyRule:
    source_module: str
    source_param: str
    target_module: str
    target_param: str
    transform_fn: Optional[Callable] = None

    def apply(self, source_value: Any) -> Any:
        if self.transform_fn:
            return self.transform_fn(source_value)
        return source_value

# class ElasticConfig:
#     def __init__(self, 
#                  elasticity_type: ElasticityType,
#                  structural_ranges: Optional[Dict[str, ElasticRange]] = None,
#                  modular_config: Optional[ModularConfig] = None,
#                  dependencies: Optional[List[DependencyRule]] = None):
#         self.elasticity_type = elasticity_type
#         self.structural_ranges = structural_ranges or {}
#         self.modular_config = modular_config
#         self.dependencies = dependencies or []


#     def __str__(self):
#         """Return a string representation of the configuration."""
#         parts = [f"ElasticConfig({self.elasticity_type.name})"]
        
#         if self.structural_ranges:
#             parts.append("\nStructural Ranges:")
#             for param, range_obj in self.structural_ranges.items():
#                 parts.append(f"  {param}: {str(range_obj)}")
        
#         if self.modular_config:
#             parts.append(f"\nModular Config: {str(self.modular_config)}")
        
#         if self.dependencies:
#             parts.append("\nDependencies:")
#             for dep in self.dependencies:
#                 parts.append(f"  {str(dep)}")
        
#         return "\n".join(parts)

@dataclass
class ElasticConfig:
    def __init__(self, 
                 structural_ranges: Dict[str, ElasticRange],
                 dependencies: Optional[List[DependencyRule]] = None,
                 init_kwargs: Optional[Dict[str, Any]] = None):  # New field for fixed arguments
        self.structural_ranges = structural_ranges
        self.dependencies = dependencies or []
        self.init_kwargs = init_kwargs or {}  # Fixed arguments for module initialization

@dataclass
class BlockElasticConfig:
    """Configuration for block-wise elasticity."""
    grouping: str          # Group identifier (e.g., "stage_1")
    min_depth: int         # Minimum number of blocks to keep
    max_depth: int         # Maximum number of blocks to keep
    blocks: List[str]      # List of block names that can be removed
    reroute_rules: Optional[Dict[str, str]] = None  # How to reconnect when blocks are removed
    
    def __str__(self):
        return f"BlockElasticConfig(group={self.grouping}, depth={self.min_depth}~{self.max_depth}, blocks={len(self.blocks)})"
