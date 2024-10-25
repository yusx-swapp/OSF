import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Optional
# from copy import deepcopy
from .elasticity import ElasticRange


class GraphIR:
    def __init__(self, model: nn.Module):
        
        self.model = model
        # Store original weights
        self.weights_dict = OrderedDict(model.state_dict())
        # Main metadata dictionary
        self.metadata_dict = OrderedDict()
        # Dictionary to store new configurations for elastic modules
        self.elastic_config_dict = OrderedDict()
        # Build the IR
        self._build_ir()

    def _build_ir(self):
        """Build metadata dictionary for user-defined modules."""
        for name, module in self.model.named_modules():
            # Skip built-in PyTorch modules
            if type(module).__module__.startswith('torch.nn'):
                continue
            
            metadata = self._create_module_metadata(name, module)
            if metadata:  # Only add if we got valid metadata
                self.metadata_dict[name] = metadata
                # Initialize empty elastic config
                self.elastic_config_dict[name] = {}

    def _create_module_metadata(self, name: str, module: nn.Module) -> Dict[str, Any]:
        """Create metadata for ResNet modules with default values."""
        import inspect
        
        # Get module's __init__ signature
        init_signature = inspect.signature(module.__class__.__init__)
        
        # Get current parameter values and defaults
        current_args = {}
        for param_name, param in init_signature.parameters.items():
            if param_name == 'self':
                continue
            
            # Try to get current value from instance
            try:
                value = getattr(module, param_name)
                current_args[param_name] = value
            except AttributeError:
                # If attribute doesn't exist, use default if available
                if param.default is not param.empty:
                    current_args[param_name] = param.default
                else:
                    current_args[param_name] = None

        metadata = {
            'module_info': {
                'type': type(module).__name__,
                'path': f"{module.__class__.__module__}.{module.__class__.__name__}",
            },
            'init_args': current_args,
            'elastic': False
        }
        return metadata

    def set_elastic_config(self, module_name: str, config: Dict[str, ElasticRange]):
        """Set elastic configuration for a module."""
        if module_name not in self.metadata_dict:
            raise KeyError(f"Module {module_name} not found")
        
        self.metadata_dict[module_name]['elastic'] = True
        self.elastic_config_dict[module_name] = config

    def sample_module_elastic_config(self, module_name: str) -> Dict[str, Any]:
        """Sample new configuration for an elastic module."""
        if not self.elastic_config_dict[module_name]:
            raise ValueError(f"No elastic config set for {module_name}")
        
        sampled_config = {}
        for param_name, range_obj in self.elastic_config_dict[module_name].items():
            sampled_config[param_name] = range_obj.sample()
        return sampled_config
    
    def sample_elastic_configs(self) -> Dict[str, Dict[str, Any]]:
        """Sample configurations for all elastic modules."""
        sampled_configs = {}
        for module_name, metadata in self.metadata_dict.items():
            if metadata['elastic'] and self.elastic_config_dict[module_name]:
                sampled_configs[module_name] = self.sample_module_elastic_config(module_name)
        return sampled_configs

    def update_elastic_config(self, module_name: str, new_config: Dict[str, Any]):
        """Update the new configuration for an elastic module."""
        if module_name not in self.metadata_dict:
            raise KeyError(f"Module {module_name} not found in metadata dictionary")
        
        if not self.metadata_dict[module_name]['elastic']:
            raise ValueError(f"Module {module_name} is not marked as elastic")
        
        self.elastic_config_dict[module_name] = new_config

    def get_module_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific module."""
        return self.metadata_dict.get(name)

    def print_metadata_dict(self, indent=2):
        """Pretty print the metadata dictionary."""
        def _format_dict(d, level=0):
            output = ""
            for key, value in d.items():
                space = " " * (level * indent)
                if isinstance(value, dict):
                    output += f"{space}{key}:\n{_format_dict(value, level + 1)}"
                else:
                    output += f"{space}{key}: {value}\n"
            return output

        print("\nMetadata Dictionary:")
        for module_name, metadata in self.metadata_dict.items():
            print(f"\n{'='*50}")
            print(f"Module: {module_name}")
            print(_format_dict(metadata))
            if self.elastic_config_dict[module_name]:
                print("\nElastic Config:")
                print(_format_dict(self.elastic_config_dict[module_name]))
    