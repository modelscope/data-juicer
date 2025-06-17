from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.optimizer.strategy import OptimizationStrategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST
from data_juicer.utils.constant import InterVars
from data_juicer.utils.registry import Registry

# Type of intermediate vars
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)
LOADED_IMAGES = Registry(InterVars.loaded_images)
LOADED_AUDIOS = Registry(InterVars.loaded_audios)
LOADED_VIDEOS = Registry(InterVars.loaded_videos)
INTER_SAMPLED_FRAMES = Registry(InterVars.sampled_frames)

ALL_INTER_VARS = [
    INTER_LINES, INTER_WORDS, LOADED_AUDIOS, LOADED_IMAGES, LOADED_VIDEOS,
    INTER_SAMPLED_FRAMES
]


class FilterFusionStrategy(OptimizationStrategy):
    """Strategy for fusing filter operations in the pipeline."""

    def __init__(self,
                 probe_results: Optional[Dict[str, Any]] = None,
                 batch_size: int = 32):
        """Initialize the filter fusion strategy.

        Args:
            probe_results: Optional dictionary containing operation speeds
            batch_size: Batch size for processing
        """
        super().__init__(name='filter_fusion')
        self.probe_results = probe_results or {}
        self.batch_size = batch_size

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply filter fusion to the pipeline AST.

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
            # Group filter operations
            filter_groups = self._group_filters(chain)

            for group in filter_groups:
                if len(group) > 1:
                    # Create fused operation
                    fused_name = f"fused_{'_'.join(n.name for n in group)}"
                    logger.info(
                        f'Fusing filter operations into {fused_name}: {[n.name for n in group]}'
                    )

                    # Create operation configs
                    op_configs = []
                    for op in group:
                        op_config = {op.name: op.config or {}}
                        op_configs.append(op_config)

                    # Create fused node
                    fused_node = OpNode(name=fused_name,
                                        op_type=OpType.FILTER,
                                        config={
                                            'general_fused_op': {
                                                'batch_size': self.batch_size,
                                                'fused_op_list': op_configs
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

    def _group_filters(self, chain: List[OpNode]) -> List[List[OpNode]]:
        """Group filter operations that can be fused together.

        Args:
            chain: List of operations in the pipeline

        Returns:
            List of filter operation groups
        """
        if not chain:
            return []

        # Group by intermediate variables
        groups: Dict[str, List[OpNode]] = {}
        for node in chain:
            if node.op_type != OpType.FILTER:
                continue

            # Get intermediate variables
            config = node.config or {}
            inter_vars = config.get('inter_vars', [node.name])

            for var in inter_vars:
                if var not in groups:
                    groups[var] = []
                groups[var].append(node)

        # Sort groups by size (largest first)
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        # Remove duplicates
        result = []
        seen = set()
        for group in sorted_groups:
            group_key = tuple(sorted(n.name for n in group))
            if group_key not in seen:
                result.append(group)
                seen.add(group_key)

        return result
