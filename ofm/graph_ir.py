import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Set, Tuple
# from copy import deepcopy
from .elasticity import ElasticRange, DependencyRule, ElasticConfig, ElasticityType
import numpy as np

class GraphIR:
    def __init__(self, model: nn.Module):
        
        self.model = model
        # Store original weights
        self.weights_dict = OrderedDict(model.state_dict())
        # Main metadata dictionary
        self.metadata_dict = OrderedDict()
        # Dictionary to store new configurations for elastic modules
        self.elastic_config_dict = OrderedDict()
        # Dependency graph
        self.dependency_graph = {}
        # Build the IR
        self._build_ir()

    def add_dependency_rule(self, rule: DependencyRule):
        """Add a dependency rule to the graph."""
        if rule.source_module not in self.dependency_graph:
            self.dependency_graph[rule.source_module] = []
        #source module is the current module, we store the rule in a hashmap
        self.dependency_graph[rule.source_module].append(rule)

    # def _get_module_groups(self) -> Dict[str, List[str]]:
    #     """Get groups of related modules."""
    #     groups = {}
    #     for module_name, config in self.elastic_config_dict.items():
    #         if config.modular_config and config.modular_config.grouping:
    #             group = config.modular_config.grouping
    #             if group not in groups:
    #                 groups[group] = []
    #             groups[group].append(module_name)
    #     return groups
    def _get_module_groups(self) -> Dict[str, List[str]]:
        """Get groups of related modules."""
        groups = {}
        for module_name, config in self.elastic_config_dict.items():
            # Check if config is a valid ElasticConfig object
            if isinstance(config, ElasticConfig) and config.modular_config:
                group = config.modular_config.grouping
                if group not in groups:
                    groups[group] = []
                groups[group].append(module_name)
        return groups
    def _validate_modular_constraints(self, removed_modules: Set[str]) -> bool:
        """Validate modular elasticity constraints."""
        groups = self._get_module_groups()
        
        for group, modules in groups.items():
            remaining = len([m for m in modules if m not in removed_modules])
            config = self.elastic_config_dict[modules[0]]
            
            if isinstance(config, ElasticConfig) and config.modular_config:
                if remaining < config.modular_config.min_depth:
                    return False
                if config.modular_config.max_depth and remaining > config.modular_config.max_depth:
                    return False
        return True

    def _reroute_connections(self, 
                           module_name: str, 
                           removed_modules: Set[str],
                           sampled_configs: Dict[str, Dict[str, Any]]):
        """Reroute connections when a module is removed."""
        if module_name not in self.dependency_graph:
            return

        for rule in self.dependency_graph[module_name]:
            if not rule.reroute_rules:
                continue

            if rule.source_module in removed_modules:
                # Update connection based on reroute rules
                input_module = rule.reroute_rules["input"]
                output_module = rule.reroute_rules["output"]
                
                # Update configurations to maintain connectivity
                if input_module in sampled_configs and output_module in sampled_configs:
                    out_channels = sampled_configs[input_module]["out_channels"]
                    sampled_configs[output_module]["in_channels"] = out_channels

    
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
                self.elastic_config_dict[name] = None

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

    def set_elastic_config(self, module_name: str, config: ElasticConfig):
        """Set elastic configuration with dependencies."""
        if module_name not in self.metadata_dict:
            raise KeyError(f"Module {module_name} not found")
        
        self.metadata_dict[module_name]['elastic'] = True
        self.elastic_config_dict[module_name] = config
        
        # Add dependencies to graph
        if config.dependencies:
            for rule in config.dependencies:
                self.add_dependency_rule(rule)
                
    def _propagate_dependencies(self, 
                              sampled_configs: Dict[str, Dict[str, Any]],
                              module_name: str):
        """Propagate configuration changes through dependencies."""
        if module_name not in self.dependency_graph:
            return

        for rule in self.dependency_graph[module_name]:
            if module_name not in sampled_configs:
                continue
                
            source_value = sampled_configs[module_name].get(rule.source_param)
            if source_value is None:
                continue
            
            # Apply transformation and update target
            target_value = rule.apply(source_value)
            
            # Create or update target module config
            if rule.target_module not in sampled_configs:
                sampled_configs[rule.target_module] = {}
            sampled_configs[rule.target_module][rule.target_param] = target_value
            
            # Recursively propagate
            self._propagate_dependencies(sampled_configs, rule.target_module)

                
    def sample_module_elastic_config(self, module_name: str) -> Dict[str, Any]:
        
        #TODO: add dependency
        return
        """Sample new configuration for an elastic module."""
        if not self.elastic_config_dict[module_name]:
            raise ValueError(f"No elastic config set for {module_name}")
        
        sampled_config = {}
        for param_name, range_obj in self.elastic_config_dict[module_name].items():
            sampled_config[param_name] = range_obj.sample()
        return sampled_config
    

    # def sample_elastic_configs(self) -> Dict[str, Dict[str, Any]]:
    #     """Sample configurations respecting dependencies."""
    #     sampled_configs = {}
        
    #     # First, sample independent modules
    #     for module_name, metadata in self.metadata_dict.items():
    #         if not metadata['elastic'] or not self.elastic_config_dict[module_name]:
    #             continue
                
    #         config = self.elastic_config_dict[module_name]
    #         sampled_config = {}
            
    #         for param_name, range_obj in config.ranges.items():
    #             sampled_config[param_name] = range_obj.sample()
            
    #         sampled_configs[module_name] = sampled_config
            
    #         # Propagate dependencies
    #         self._propagate_dependencies(sampled_configs, module_name)
        
    #     return sampled_configs

    def sample_elastic_configs(self) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
        """Sample configurations and determine which modules to remove."""
        sampled_configs = {}
        removed_modules = set()
        
        # First, decide which modules to remove
        for module_name, config in self.elastic_config_dict.items():
            # Check if config is a valid ElasticConfig object
            if (isinstance(config, ElasticConfig) and 
                config.modular_config and 
                config.modular_config.removable and 
                np.random.random() > 0.5):  # 50% chance to remove
                removed_modules.add(module_name)
        
        # Validate modular constraints
        if not self._validate_modular_constraints(removed_modules):
            return self.sample_elastic_configs()  # Changed from sample_all_elastic_configs
        
        # Sample structural configs for remaining modules
        for module_name, config in self.elastic_config_dict.items():
            if module_name in removed_modules:
                continue
                
            if isinstance(config, ElasticConfig) and config.elasticity_type == ElasticityType.STRUCTURAL:
                sampled_config = {}
                for param_name, range_obj in config.structural_ranges.items():
                    sampled_config[param_name] = range_obj.sample()
                sampled_configs[module_name] = sampled_config
        
        # Handle rerouting for removed modules
        for module_name in removed_modules:
            self._reroute_connections(module_name, removed_modules, sampled_configs)
        
        return sampled_configs, removed_modules
    def update_elastic_config(self, module_name: str, new_config: Dict[str, Any]):
        #TODO add should update dependency as well
        return 
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
    def print_elastic_configs(self) -> str:
        """Generate a mermaid diagram showing all elastic configurations and their relationships."""
        mermaid = ["graph TD"]
        
        # Add nodes for each module with elastic config
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
                
            # Create node ID (replace dots with underscore for mermaid compatibility)
            node_id = module_name.replace(".", "_")
            
            # Add module node
            if config.elasticity_type == ElasticityType.STRUCTURAL:
                ranges_str = "<br>".join(f"{p}: {r}" for p, r in config.structural_ranges.items())
                mermaid.append(f'    {node_id}["{module_name}<br>{ranges_str}"]')
            else:
                mermaid.append(f'    {node_id}["{module_name}<br>Modular"]')
            
            # Add group subgraph if module has modular config
            if config.modular_config and config.modular_config.grouping:
                group = config.modular_config.grouping
                mermaid.append(f'    subgraph {group}')
                mermaid.append(f'        {node_id}')
                mermaid.append('    end')
        
        # Add dependency edges
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
                
            source_id = module_name.replace(".", "_")
            for dep in config.dependencies:
                target_id = dep.target_module.replace(".", "_")
                label = f"{dep.source_param}->{dep.target_param}"
                mermaid.append(f'    {source_id} -->|"{label}"| {target_id}')
        print("\n".join(mermaid))
        return "\n".join(mermaid)