import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yaml


class OpType(Enum):
    """Types of operations in the pipeline."""
    ROOT = 'root'
    MAPPER = 'mapper'
    FILTER = 'filter'
    DEDUPLICATOR = 'deduplicator'
    SELECTOR = 'selector'
    GROUPER = 'grouper'
    AGGREGATOR = 'aggregator'


@dataclass
class OpNode:
    """Node in the pipeline AST representing an operation."""
    name: str
    op_type: OpType
    config: Dict[str, Any]
    children: List['OpNode'] = None
    parent: Optional['OpNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: 'OpNode'):
        """Add a child node to this operation."""
        child.parent = self
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the node to a dictionary representation."""
        return {
            'name': self.name,
            'type': self.op_type.value,
            'config': self.config,
            'children': [child.to_dict() for child in self.children]
        }


class PipelineAST:
    """Abstract Syntax Tree for a Data-Juicer pipeline."""

    def __init__(self):
        self.root = None
        self._op_type_map = {
            'mapper': OpType.MAPPER,
            'filter': OpType.FILTER,
            'deduplicator': OpType.DEDUPLICATOR,
            'selector': OpType.SELECTOR,
            'grouper': OpType.GROUPER,
            'aggregator': OpType.AGGREGATOR
        }

        # Operation dependencies and optimization rules
        self._op_dependencies = {
            OpType.FILTER: {OpType.MAPPER},  # Filters can depend on mappers
            OpType.DEDUPLICATOR:
            {OpType.MAPPER,
             OpType.FILTER},  # Deduplicators can depend on mappers and filters
            OpType.SELECTOR:
            {OpType.MAPPER, OpType.FILTER,
             OpType.DEDUPLICATOR},  # Selectors can depend on all previous ops
            OpType.GROUPER: {
                OpType.MAPPER, OpType.FILTER, OpType.DEDUPLICATOR,
                OpType.SELECTOR
            },  # Groupers can depend on all previous ops
            OpType.AGGREGATOR:
            {OpType.GROUPER}  # Aggregators can only depend on groupers
        }

    def _get_op_type(self, op_name: str) -> OpType:
        """Determine the operation type from its name."""
        for suffix, op_type in self._op_type_map.items():
            if op_name.endswith(f'_{suffix}'):
                return op_type
        return OpType.MAPPER  # Default to mapper if type cannot be determined

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the AST from a configuration dictionary."""
        if 'process' not in config:
            raise ValueError("Configuration must contain a 'process' field")

        process_list = config['process']
        if not process_list:
            return

        # Create root node
        self.root = OpNode(
            name='root',
            op_type=OpType.ROOT,  # Root is a special type
            config={})

        # Build tree following the order in process_list
        current_node = self.root
        for op_config in process_list:
            op_name, op_args = list(op_config.items())[0]
            op_type = self._get_op_type(op_name)

            new_node = OpNode(name=op_name, op_type=op_type, config=op_args)
            current_node.add_child(new_node)
            current_node = new_node

    def build_from_yaml(self, yaml_path: str) -> None:
        """Build the AST from a YAML configuration file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        self.build_from_config(config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the AST to a dictionary representation."""
        if not self.root:
            return {}
        return self.root.to_dict()

    def visualize(self) -> str:
        """Generate a string representation of the AST for visualization."""
        if not self.root:
            return 'Empty pipeline'

        def _visualize_node(node: OpNode,
                            level: int = 0,
                            is_last: bool = True) -> str:
            indent = '  ' * level
            prefix = '└── ' if is_last else '├── '
            result = f'{indent}{prefix}{node.name} ({node.op_type.value})\n'

            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                result += _visualize_node(child, level + 1, is_last_child)
            return result

        return 'Pipeline:\n' + _visualize_node(self.root, 0, True)

    def get_operation_chain(self) -> List[OpNode]:
        """Get the linear chain of operations in the pipeline."""
        if not self.root:
            return []

        chain = []
        current = self.root
        while current.children:
            current = current.children[0]
            chain.append(current)
        return chain

    def _validate_dependencies(self) -> List[Tuple[OpNode, OpNode]]:
        """Validate operation dependencies and return invalid dependencies."""
        invalid_deps = []
        chain = self.get_operation_chain()

        for i, node in enumerate(chain):
            if node.op_type in self._op_dependencies:
                allowed_deps = self._op_dependencies[node.op_type]
                for j in range(i):
                    if chain[j].op_type not in allowed_deps:
                        invalid_deps.append((chain[j], node))

        return invalid_deps

    def _optimize_operation_order(self) -> None:
        """Optimize the order of operations based on dependencies and efficiency."""
        chain = self.get_operation_chain()
        if not chain:
            return

        # Group operations by type
        op_groups: Dict[OpType,
                        List[OpNode]] = {op_type: []
                                         for op_type in OpType}
        for node in chain:
            op_groups[node.op_type].append(node)

        # Rebuild the tree with optimized order
        self.root = OpNode(name='root', op_type=OpType.MAPPER, config={})
        current = self.root

        # Process operations in order: Mapper -> Filter -> Deduplicator -> Selector -> Grouper -> Aggregator
        for op_type in [
                OpType.MAPPER, OpType.FILTER, OpType.DEDUPLICATOR,
                OpType.SELECTOR, OpType.GROUPER, OpType.AGGREGATOR
        ]:
            for node in op_groups[op_type]:
                new_node = OpNode(name=node.name,
                                  op_type=node.op_type,
                                  config=node.config)
                current.add_child(new_node)
                current = new_node

    def _merge_compatible_operations(self) -> None:
        """Merge compatible operations to reduce pipeline complexity."""
        chain = self.get_operation_chain()
        if not chain:
            return

        i = 0
        while i < len(chain) - 1:
            current = chain[i]
            next_op = chain[i + 1]

            # Check if operations can be merged
            if (current.op_type == next_op.op_type
                    and current.op_type in [OpType.MAPPER, OpType.FILTER]):
                # Merge configurations
                merged_config = {**current.config, **next_op.config}
                merged_node = OpNode(
                    name=f'merged_{current.name}_{next_op.name}',
                    op_type=current.op_type,
                    config=merged_config)

                # Replace the two nodes with the merged node
                if current.parent:
                    current.parent.children.remove(current)
                    current.parent.add_child(merged_node)
                    if next_op.parent:
                        next_op.parent.children.remove(next_op)
                        merged_node.add_child(
                            next_op.children[0] if next_op.children else None)

                chain[i] = merged_node
                chain.pop(i + 1)
            else:
                i += 1

    def optimize(self) -> None:
        """Optimize the pipeline AST based on operation dependencies and resource usage."""
        if not self.root:
            return

        # Validate dependencies
        invalid_deps = self._validate_dependencies()
        if invalid_deps:
            print('Warning: Invalid operation dependencies found:')
            for dep, node in invalid_deps:
                print(f'  {dep.name} -> {node.name}')

        # Optimize operation order
        self._optimize_operation_order()

        # Merge compatible operations
        self._merge_compatible_operations()


if __name__ == '__main__':
    import os

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Build and visualize pipeline AST from config file')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/data_juicer_recipes/pile-philpaper-refine.yaml',
        help='Path to the pipeline configuration file (YAML)')
    parser.add_argument(
        '--probe-results',
        type=str,
        help='Path to probe results file (YAML) containing operation speeds')
    parser.add_argument('--optimize',
                        action='store_true',
                        help='Apply optimization strategies to the pipeline')

    args = parser.parse_args()

    # Get absolute path to config file
    config_path = os.path.abspath(args.config)
    print(f'Using config file: {config_path}')

    # Load and process config
    config = yaml.safe_load(open(config_path, 'r'))

    # Build initial AST
    ast = PipelineAST()
    ast.build_from_config(config)
    print('\nOriginal Pipeline:')
    print(ast.visualize())

    # Apply optimization if requested
    if args.optimize:
        from data_juicer.core.optimizer.filter_fusion_strategy import \
            OpFusionStrategy
        from data_juicer.core.optimizer.optimizer import PipelineOptimizer

        # Load probe results if provided
        probe_results = None
        if args.probe_results:
            probe_path = os.path.abspath(args.probe_results)
            print(f'\nUsing probe results from: {probe_path}')
            probe_results = yaml.safe_load(open(probe_path, 'r'))

        # Create optimizer with op fusion strategy
        optimizer = PipelineOptimizer(
            [OpFusionStrategy(probe_results=probe_results)])

        # Apply optimization
        optimized_ast = optimizer.optimize(ast)

        print('\nOptimized Pipeline:')
        print(optimized_ast.visualize())
