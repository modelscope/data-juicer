from typing import List

from loguru import logger

from data_juicer.core.optimizer.strategy import OptimizationStrategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST


class MapperFusionStrategy(OptimizationStrategy):
    """Strategy for fusing mapper operations in the pipeline."""

    def __init__(self, batch_size: int = 32):
        """Initialize the mapper fusion strategy.

        Args:
            batch_size: Batch size for processing
        """
        super().__init__(name='mapper_fusion')
        self.batch_size = batch_size

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply mapper fusion to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST
        """
        if not ast.root:
            return ast

        # Create a new AST
        new_ast = PipelineAST()
        new_ast.root = OpNode(name='root', op_type=OpType.ROOT, config={})

        # Get all unique operation chains
        op_chains = self._get_unique_op_chains(ast.root)

        # Process each chain
        current = new_ast.root
        for chain in op_chains:
            # Group mapper operations
            mapper_groups = self._group_mappers(chain)

            for group in mapper_groups:
                if len(group) > 1:
                    # Create fused operation
                    fused_name = f"fused_{'_'.join(n.name for n in group)}"
                    logger.info(
                        f'Fusing mapper operations into {fused_name}: {[n.name for n in group]}'
                    )

                    # Create fused node using FusedMapper
                    fused_node = OpNode(name=fused_name,
                                        op_type=OpType.MAPPER,
                                        config={
                                            'fused_mapper': {
                                                'name':
                                                fused_name,
                                                'fused_mappers':
                                                [op.name for op in group],
                                                'batch_size':
                                                self.batch_size
                                            }
                                        })
                    current.add_child(fused_node)
                    current = fused_node
                else:
                    # Keep single operations as is
                    new_node = OpNode(name=group[0].name,
                                      op_type=group[0].op_type,
                                      config=group[0].config or {})
                    current.add_child(new_node)
                    current = new_node

        return new_ast

    def _get_unique_op_chains(self, node: OpNode) -> List[List[OpNode]]:
        """Get unique chains of operations from the tree.

        Args:
            node: Root node of the tree

        Returns:
            List of unique operation chains
        """
        chains = []
        seen_chains = set()

        def traverse(current: OpNode, chain: List[OpNode]):
            if not current.children:
                # End of chain, check if we've seen this sequence before
                chain_key = tuple(n.name for n in chain)
                if chain_key not in seen_chains:
                    chains.append(chain.copy())
                    seen_chains.add(chain_key)
                return

            for child in current.children:
                chain.append(child)
                traverse(child, chain)
                chain.pop()

        traverse(node, [])
        return chains

    def _group_mappers(self, chain: List[OpNode]) -> List[List[OpNode]]:
        """Group mapper operations that can be fused together.

        Args:
            chain: List of operations in the pipeline

        Returns:
            List of mapper operation groups
        """
        groups = []
        current_group = []

        for node in chain:
            if node.op_type != OpType.MAPPER:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([node])
            else:
                if not current_group:
                    current_group.append(node)
                else:
                    # Check if current mapper can be fused with the group
                    if self._can_fuse_with_group(node, current_group):
                        current_group.append(node)
                    else:
                        groups.append(current_group)
                        current_group = [node]

        if current_group:
            groups.append(current_group)

        return groups

    def _can_fuse_with_group(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if a mapper can be fused with a group.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if the operation can be fused with the group
        """
        # Check dependencies
        for op in group:
            if self._has_dependency(node, op) or self._has_dependency(
                    op, node):
                return False

        return True

    def _has_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if op1 depends on op2.

        Args:
            op1: First operation
            op2: Second operation

        Returns:
            True if op1 depends on op2
        """
        # Get intermediate variables
        op1_vars = set(op1.config.get('inter_vars', []))
        op2_vars = set(op2.config.get('inter_vars', []))

        # Check if op1 uses any variables produced by op2
        return bool(op1_vars & op2_vars)
