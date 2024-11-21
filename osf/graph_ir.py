import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Set, Tuple
# from copy import deepcopy
from .elasticity import ElasticRange, DependencyRule, ElasticConfig, ElasticityType, BlockElasticConfig
import numpy as np
from copy import deepcopy
import torch
class GraphIR:
    """Graph-based Intermediate Representation for weight-sharing neural architecture search.
    
    This class manages the network topology, module configurations, and weight sharing
    for architecture-agnostic neural network transformation and training.
    """
    
    #############################################
    # Initialization and Construction
    #############################################
    
    def __init__(self, model: nn.Module):
        
        self.supernet = model
        # Store original weights
        self.weights_dict = OrderedDict(self.supernet.state_dict())
        # Main metadata dictionary
        self.metadata_dict = OrderedDict()
        # Dictionary to store new configurations for elastic modules
        self.elastic_config_dict = OrderedDict()
        # Dependency graph
        self.dependency_graph = {}
        # Build the IR
        self._build_ir()

    def _build_ir(self):
        """Build metadata dictionary for user-defined modules."""
        for name, module in self.supernet.named_modules():

            
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

    #############################################
    # Elastic Configuration Management 
    #############################################

    def set_elastic_config(self, module_name: str, config: ElasticConfig):
        """Set elastic configuration with dependencies."""
        if module_name not in self.metadata_dict:
            # self.metadata_dict[module_name] = {}
            # self.elastic_config_dict[module_name] = {}
            raise KeyError(f"Module {module_name} not found")
        
        self.metadata_dict[module_name]['elastic'] = True
        self.elastic_config_dict[module_name] = config
        
        # Add dependencies to graph
        if config.dependencies:
            for rule in config.dependencies:
                if rule:
                    self.add_dependency_rule(rule)
    def set_block_elastic_config(self, group_name: str, config: BlockElasticConfig):
        """Set block elastic configuration for a group of blocks."""
        self.block_elastic_config_dict[group_name] = config
     
    def update_elastic_config(self, module_name: str, new_config: Dict[str, Any]):
        #TODO add should update dependency as well
        return 
        """Update the new configuration for an elastic module."""
        if module_name not in self.metadata_dict:
            raise KeyError(f"Module {module_name} not found in metadata dictionary")
        
        if not self.metadata_dict[module_name]['elastic']:
            raise ValueError(f"Module {module_name} is not marked as elastic")
        
        self.elastic_config_dict[module_name] = new_config
    #############################################
    # Subnet Management & Creation
    #############################################

    def create_subnet(self, sampled_configs: Dict[str, Dict[str, Any]]) -> nn.Module:
        """Create a subnet based on the sampled elastic configurations."""
        subnet = deepcopy(self.supernet)
       
        def fix_module_children(new_mod, orig_mod):
            for name, child in new_mod.named_children():
                orig_child = getattr(orig_mod, name)
                if type(child) != type(orig_child):
                    setattr(new_mod, name, deepcopy(orig_child))
        for module_name, config in sampled_configs.items():
            names = module_name.split('.')
            curr_module = subnet
            for name in names[:-1]:
                curr_module = getattr(curr_module, name)
            
            original_module = getattr(curr_module, names[-1])
            module_class = type(original_module)
            
            # Create new module with new config
            init_args = self.metadata_dict[module_name]['init_args'].copy()
            init_args.update(config)
            new_module = module_class(**init_args)
            

            # Fix children modules right after creation            
            fix_module_children(new_module, original_module)
            setattr(curr_module, names[-1], new_module)
        
        self._copy_weights(subnet)
        return subnet
    def create_subnet__(self, sampled_configs: Dict[str, Dict[str, Any]]) -> nn.Module:
        """Create a subnet based on the sampled elastic configurations.
        
        Args:
            sampled_configs: Dictionary mapping module names to their sampled configurations
                            Example: {'resnet.encoder.stages.0.layers.0': {'in_channels': 64, 'out_channels': 128}}
        
        Returns:
            nn.Module: A new model instance with the sampled configurations
        """
        # Create a deep copy of the original model
        subnet = deepcopy(self.supernet)
        
        # For each module in the sampled configs
        for module_name, config in sampled_configs.items():
            # Get the module from subnet using the module name
            names = module_name.split('.')
            curr_module = subnet
            for name in names[:-1]:
                curr_module = getattr(curr_module, name)
            
            # Get original module to access its class and default arguments
            original_module = getattr(curr_module, names[-1])
            module_class = type(original_module)
            
        
            # Get initialization arguments from metadata
            # init_args = self.metadata_dict[module_name]['init_args'].copy()
            # init_args.update(config)
               
            new_module = module_class(**config)
            
            # Replace the module in subnet
            setattr(curr_module, names[-1], new_module)
        
        self._copy_weights(subnet)
        
        return subnet
    def _copy_weights(self, subnet: nn.Module) -> None:
        """Copy weights from supernet to subnet.
        Only copies overlapping parts of weights where subnet dimensions are smaller.
        
        Args:
            subnet: The target subnet with potentially smaller dimensions
        """
        # Get source and target state dicts
        source_state_dict = self.supernet.state_dict()
        target_state_dict = subnet.state_dict()
        
        # Create new state dict for subnet
        new_state_dict = OrderedDict()
    
        for key, target_tensor in target_state_dict.items():
            source_tensor = source_state_dict[key]
            
            if target_tensor.dim() == source_tensor.dim():
                # Check if all dimensions in subnet are smaller or equal
                if all(t_dim <= s_dim 
                    for t_dim, s_dim in zip(target_tensor.shape, source_tensor.shape)):
                    # Create slice objects for each dimension
                    slices = tuple(
                        slice(0, min(t_dim, s_dim))
                        for t_dim, s_dim in zip(target_tensor.shape, source_tensor.shape)
                    )
                    new_state_dict[key] = source_tensor[slices].clone()
                else:
                    print(f"module {key} dim not match, target shape:{target_tensor.shape}, dim: {target_tensor.dim()}; source shape:{source_tensor.shape}, dim: {source_tensor.dim()}")
                    print("Weights not copied, keep original")
                    new_state_dict[key] = target_tensor.clone()  # Keep original if dimensions don't match
            else:
                # If dimensions don't match, copy the tensor as is
                # new_state_dict[key] = source_tensor
                print(f"module {key} dim not match, target shape:{target_tensor.shape}, dim: {target_tensor.dim()}; source shape:{source_tensor.shape}, dim: {source_tensor.dim()}")
                new_state_dict[key] = source_tensor.clone()
                continue
        # Load weights into subnet
        subnet.load_state_dict(new_state_dict)
    
    #############################################
    # Sampling & Configuration Generation
    #############################################

    def sample_elastic_configs(self) -> Dict[str, Dict[str, Any]]:
        """Sample configurations respecting dependencies."""
        sampled_configs = {}
        
        # First sample all configurations
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
                
            sampled_config = {}
            for param_name, range_obj in config.structural_ranges.items():
                sampled_config[param_name] = range_obj.sample()
            
            # Add fixed initialization arguments
            sampled_config.update(config.init_kwargs)
            sampled_configs[module_name] = sampled_config
                
        # Then apply all dependencies to correct the configurations
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
            
            self._propagate_dependencies(sampled_configs, module_name)
        
        return sampled_configs

    def sample_elastic_configs__(self) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
        """Sample configurations and determine which modules to remove."""
        sampled_configs = {}
        removed_modules = set()
        
        # First build a dependency order
        order = []
        visited = set()
        
        def visit(module_name: str):
            if module_name in visited:
                return
            visited.add(module_name)
            
            # First visit any modules this module depends on
            config = self.elastic_config_dict.get(module_name)
            if isinstance(config, ElasticConfig) and config.dependencies:
                for dep in config.dependencies:
                    # If this module's parameter depends on another module,
                    # process that module first
                    if dep.source_module == module_name:
                        visit(dep.target_module)
            
            order.append(module_name)
        
        # Build processing order
        for module_name in self.elastic_config_dict.keys():
            visit(module_name)
        
        print(f"Processing order: {order}")  # Debug print
        
        # Now sample in order
        for module_name in order:
            if module_name in removed_modules:
                continue
                
            config = self.elastic_config_dict.get(module_name)
            if not isinstance(config, ElasticConfig):
                continue
                
            # Check if this module's parameters are determined by dependencies
            determined_params = {}
            if module_name in sampled_configs:
                determined_params = sampled_configs[module_name]
            
            # Sample only parameters that aren't determined by dependencies
            sampled_config = {}
            for param_name, range_obj in config.structural_ranges.items():
                if param_name not in determined_params:
                    sampled_config[param_name] = range_obj.sample()
                else:
                    sampled_config[param_name] = determined_params[param_name]
            
            sampled_configs[module_name] = sampled_config
            
            # Propagate this configuration
            if config.dependencies:
                for dep in config.dependencies:
                    if dep.source_module == module_name:  # only propagate if this is the source
                        source_value = sampled_config[dep.source_param]
                        target_value = dep.transform_fn(source_value) if dep.transform_fn else source_value
                        
                        if dep.target_module not in sampled_configs:
                            sampled_configs[dep.target_module] = {}
                        sampled_configs[dep.target_module][dep.target_param] = target_value
        
        print("Final configs:", sampled_configs)  # Debug print
        return sampled_configs, removed_modules
    def sample_max_elastic_config(self) -> Dict[str, Dict[str, Any]]:
        """Sample configuration with maximum values for elastic parameters."""
        sampled_configs = {}
        
        # Sample maximum values for each module
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
                
            sampled_config = {}
            for param_name, range_obj in config.structural_ranges.items():
                sampled_config[param_name] = range_obj.max_val
            # Add fixed initialization arguments
            sampled_config.update(config.init_kwargs)
            sampled_configs[module_name] = sampled_config
        
        # Apply dependencies
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
            self._propagate_dependencies(sampled_configs, module_name)
        
        return sampled_configs

    def sample_min_elastic_config(self) -> Dict[str, Dict[str, Any]]:
        """Sample configuration with minimum values for elastic parameters."""
        sampled_configs = {}
        
        # Sample minimum values for each module
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
                
            sampled_config = {}
            for param_name, range_obj in config.structural_ranges.items():
                sampled_config[param_name] = range_obj.min_val
            # Add fixed initialization arguments
            sampled_config.update(config.init_kwargs)
            sampled_configs[module_name] = sampled_config
        
        # Apply dependencies
        for module_name, config in self.elastic_config_dict.items():
            if not isinstance(config, ElasticConfig):
                continue
            self._propagate_dependencies(sampled_configs, module_name)
        
        return sampled_configs
    
    def sample_block_config(self) -> Tuple[Set[str], Set[str]]:
        """Sample which blocks to keep and remove.
        
        Returns:
            Tuple[Set[str], Set[str]]: (kept_blocks, removed_blocks)PP
        """
        kept_blocks = set()
        removed_blocks = set()
        
        # Process each group
        for group_name, config in self.block_elastic_config_dict.items():
            available_blocks = config.blocks.copy()
            
            # Randomly decide how many blocks to keep (between min and max depth)
            num_blocks = len(available_blocks)
            num_keep = np.random.randint(config.min_depth, min(config.max_depth, num_blocks) + 1)
            
            # Randomly select blocks to keep
            blocks_to_keep = set(np.random.choice(
                available_blocks, 
                size=num_keep, 
                replace=False
            ))
            
            # Update kept and removed sets
            kept_blocks.update(blocks_to_keep)
            removed_blocks.update(set(available_blocks) - blocks_to_keep)
        
        return kept_blocks, removed_blocks
    #############################################
    # Weight & Gradient Management
    #############################################

    def apply_subnet_grads(self, subnet: nn.Module):
        """Apply gradients from trained subnet to supernet.
        
        Two possible approaches:
        1. Gradient copying (if we need gradient accumulation):
        """
        #make the subnet and supernet in the same device
        subnet = subnet.to(self.supernet.device)
        for super_param, subnet_param in zip(self.supernet.parameters(), subnet.parameters()):
            if subnet_param.grad is not None:
                # Handle different tensor sizes
                if super_param.shape != subnet_param.shape:
                    # Create zero grad of supernet shape if not exists
                    if super_param.grad is None:
                        super_param.grad = torch.zeros_like(super_param)
                    # Copy gradients to corresponding positions
                    slices = tuple(slice(0, dim) for dim in subnet_param.shape)
                    super_param.grad[slices].copy_(subnet_param.grad)
                else:
                    # Direct copy for same shape
                    if super_param.grad is None:
                        super_param.grad = subnet_param.grad.clone()
                    else:
                        super_param.grad.copy_(subnet_param.grad)

        
    def accumulate_subnet_grads(self, subnets: List[nn.Module]):
        """Fuse gradients from multiple parallel trained subnets to supernet.
        
        Similar to weight fusion, but operates on gradients instead of weights.
        Gradients in overlapping regions are averaged based on coverage count.
        
        Args:
            subnets: List of trained subnets with computed gradients
        """
        # Initialize gradient accumulation tensors and coverage counters
        accum_grads = {}
        coverage_count = {}
        
        # Initialize accumulation tensors for each parameter
        for name, param in self.supernet.named_parameters():
            accum_grads[name] = torch.zeros_like(param.data)
            coverage_count[name] = torch.zeros_like(param.data, dtype=torch.int)
            # Ensure supernet gradients are initialized
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
        
        # Accumulate gradients from each subnet
        for subnet in subnets:
            subnet = subnet.to(self.supernet.device)
            
            for (super_name, super_param), (subnet_name, subnet_param) in zip(
                self.supernet.named_parameters(), subnet.named_parameters()):
                
                if subnet_param.grad is not None:
                    # Get slices for this subnet's parameter region
                    slices = tuple(slice(0, dim) for dim in subnet_param.shape)
                    
                    # Accumulate gradients
                    accum_grads[super_name][slices] += subnet_param.grad.data
                    # Increment coverage counter for this region
                    coverage_count[super_name][slices] += 1
        
        # Average accumulated gradients by coverage count and update supernet gradients
        for name, param in self.supernet.named_parameters():
            # Avoid division by zero
            mask = coverage_count[name] > 0
            if mask.any():
                param.grad.data[mask] = accum_grads[name][mask] / coverage_count[name][mask]   
    
    def fuse_subnet_weights(self, subnet: nn.Module):
        """Fuse trained subnet weights back to supernet.
        
        This function updates the supernet's weights with the trained subnet's weights
        for corresponding regions where they overlap.
        
        Args:
            subnet: The trained subnet with updated weights
        """
        subnet = subnet.to(self.supernet.device)
        
        for (super_name, super_param), (subnet_name, subnet_param) in zip(
            self.supernet.named_parameters(), subnet.named_parameters()):
            
            # If shapes match, direct update
            if super_param.shape == subnet_param.shape:
                super_param.data.copy_(subnet_param.data)
            else:
                # For different shapes, update the overlapping region
                slices = tuple(slice(0, dim) for dim in subnet_param.shape)
                super_param.data[slices].copy_(subnet_param.data)
              
              
    def accumulate_subnet_weights(self, subnets: List[nn.Module]):
        """Fuse weights from multiple parallel trained subnets to supernet.
        
        For overlapping regions, weights are averaged based on the number of 
        subnets that cover each region.
        
        Args:
            subnets: List of trained subnets
        """
        # Initialize weight accumulation tensors and coverage counters
        accum_weights = {}
        coverage_count = {}
        
        # Initialize accumulation tensors for each parameter
        for name, param in self.supernet.named_parameters():
            accum_weights[name] = torch.zeros_like(param.data)
            coverage_count[name] = torch.zeros_like(param.data, dtype=torch.int)
        
        # Accumulate weights from each subnet
        for subnet in subnets:
            subnet = subnet.to(self.supernet.device)
            
            for (super_name, super_param), (subnet_name, subnet_param) in zip(
                self.supernet.named_parameters(), subnet.named_parameters()):
                
                # Get slices for this subnet's parameter region
                slices = tuple(slice(0, dim) for dim in subnet_param.shape)
                
                # Accumulate weights
                accum_weights[super_name][slices] += subnet_param.data
                # Increment coverage counter for this region
                coverage_count[super_name][slices] += 1
        
        # Average accumulated weights by coverage count and update supernet
        for name, param in self.supernet.named_parameters():
            # Avoid division by zero
            mask = coverage_count[name] > 0
            if mask.any():
                param.data[mask] = accum_weights[name][mask] / coverage_count[name][mask]
                
    #############################################
    # Dependency Management
    #############################################

    def add_dependency_rule(self, rule: DependencyRule):
        """Add a dependency rule to the graph."""
        if rule.source_module not in self.dependency_graph:
            self.dependency_graph[rule.source_module] = []
        #source module is the current module, we store the rule in a hashmap
        self.dependency_graph[rule.source_module].append(rule)
    
    def _propagate_dependencies(self, sampled_configs: Dict[str, Dict[str, Any]], module_name: str):
        """Propagate configuration changes through dependencies."""
        if module_name not in self.dependency_graph:
            return

        for rule in self.dependency_graph[module_name]:
            source_value = sampled_configs[module_name].get(rule.source_param)
            if source_value is None:
                continue
            
            # Apply transformation and update target
            target_value = rule.apply(source_value)
            
            # Create or update target module config
            if rule.target_module not in sampled_configs:
                sampled_configs[rule.target_module] = {}
            sampled_configs[rule.target_module][rule.target_param] = target_value

    def _propagate_dependencies__(self, sampled_configs: Dict[str, Dict[str, Any]], module_name: str):
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
            if rule.transform_fn:
                target_value = rule.transform_fn(source_value)
            else:
                target_value = source_value  # Direct copy if no transform_fn
            
            # Create or update target module config
            if rule.target_module not in sampled_configs:
                sampled_configs[rule.target_module] = {}
            sampled_configs[rule.target_module][rule.target_param] = target_value
            
            # Recursively propagate
            self._propagate_dependencies(sampled_configs, rule.target_module)

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

    #############################################
    # Utility & Debug Functions
    #############################################

    def get_module_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific module."""
        return self.metadata_dict.get(name)

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
    
    