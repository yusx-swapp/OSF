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

@dataclass
class ModularConfig:
    """Configuration for modular elasticity."""
    removable: bool = False  # Whether module can be removed
    min_depth: int = 1      # Minimum number of modules to keep
    max_depth: int = None   # Maximum number of modules (None means no limit)
    grouping: Optional[str] = None  # Group identifier for related modules

@dataclass
class DependencyRule:
    """Define how parameters depend on each other."""
    source_module: str  # Source module name
    source_param: str   # Source parameter name
    target_module: str  # Target module name
    target_param: str   # Target parameter name
    transform_fn: Optional[Callable] = None  # Optional transformation function
    reroute_rules: Optional[Dict[str, str]] = None  # Maps source outputs to target inputs when module is removed

    
    def apply(self, source_value: Any) -> Any:
        """Apply transformation to source value."""
        if self.transform_fn:
            return self.transform_fn(source_value)
        return source_value

class ElasticConfig:
    def __init__(self, 
                 elasticity_type: ElasticityType,
                 structural_ranges: Optional[Dict[str, ElasticRange]] = None,
                 modular_config: Optional[ModularConfig] = None,
                 dependencies: Optional[List[DependencyRule]] = None):
        self.elasticity_type = elasticity_type
        self.structural_ranges = structural_ranges or {}
        self.modular_config = modular_config
        self.dependencies = dependencies or []