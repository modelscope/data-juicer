from typing import Any, Dict, List, Optional

from loguru import logger

from data_juicer.core.optimizer.strategy import OptimizationStrategy
from data_juicer.core.pipeline_ast import OpNode, OpType, PipelineAST
from data_juicer.utils.constant import InterVars, StatsKeys
from data_juicer.utils.registry import Registry

# Type of intermediate vars
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)
LOADED_IMAGES = Registry(InterVars.loaded_images)
LOADED_AUDIOS = Registry(InterVars.loaded_audios)
LOADED_VIDEOS = Registry(InterVars.loaded_videos)
INTER_SAMPLED_FRAMES = Registry(InterVars.sampled_frames)

ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_AUDIOS, LOADED_IMAGES, LOADED_VIDEOS, INTER_SAMPLED_FRAMES]


class FilterFusionStrategy(OptimizationStrategy):
    """Strategy for fusing filter operations in the pipeline."""

    def __init__(
        self, probe_results: Optional[Dict[str, Any]] = None, analyzer_insights: Optional[Dict[str, Any]] = None
    ):
        """Initialize the filter fusion strategy.

        Args:
            probe_results: Optional dictionary containing operation speeds
            analyzer_insights: Optional dictionary containing dataset analysis insights
        """
        super().__init__(name="filter_fusion")
        self.probe_results = probe_results or {}
        self.analyzer_insights = analyzer_insights or {}

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
        new_ast.root = OpNode(name="root", op_type=OpType.ROOT, config={})

        # Get all unique operation chains
        op_chains = self._get_unique_op_chains(ast.root)

        # Process each chain
        current = new_ast.root
        for chain in op_chains:
            # Group filter operations with analyzer insights
            filter_groups = self._group_filters_with_insights(chain)

            for group in filter_groups:
                if len(group) > 1:
                    # Create fused operation with clean naming
                    fused_name = "fused_filter"
                    detailed_ops = [n.name for n in group]
                    logger.info(f"Fusing filter operations into {fused_name}: {detailed_ops}")

                    # Create operation configs
                    op_configs = []
                    for op in group:
                        op_config = {op.name: op.config or {}}
                        op_configs.append(op_config)

                    # Create fused node
                    fused_node = OpNode(
                        name=fused_name,
                        op_type=OpType.FILTER,
                        config={
                            "general_fused_op": {
                                "fused_op_list": op_configs,
                                "detailed_ops": detailed_ops,  # For display purposes
                            }
                        },
                    )
                    current.add_child(fused_node)
                    current = fused_node
                else:
                    # Keep single operations as is
                    new_node = OpNode(name=group[0].name, op_type=group[0].op_type, config=group[0].config or {})
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

    def _group_filters_with_insights(self, chain: List[OpNode]) -> List[List[OpNode]]:
        """Group filter operations using analyzer insights for better decisions.

        Args:
            chain: List of operations in the pipeline

        Returns:
            List of filter operation groups
        """
        groups = []
        current_group = []

        for node in chain:
            if not PipelineAST.is_filter_op(node):
                # If we encounter a non-filter, finalize current group
                if current_group:
                    groups.append(current_group)
                    current_group = []
                # Add the non-filter node as a separate group
                groups.append([node])
            else:
                # This is a filter node
                if not current_group:
                    # Start a new group
                    current_group = [node]
                else:
                    # Check if current filter can be fused with the group using insights
                    if self._can_fuse_with_group_insights(node, current_group):
                        current_group.append(node)
                    else:
                        # Finalize current group and start a new one
                        groups.append(current_group)
                        current_group = [node]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _can_fuse_with_group_insights(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if a filter can be fused with a group using analyzer insights.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if the operation can be fused with the group
        """
        # Basic dependency check
        for op in group:
            if self._has_dependency(node, op) or self._has_dependency(op, node):
                return False

        # Use analyzer insights for advanced decisions
        if self.analyzer_insights:
            return self._analyzer_based_fusion_decision(node, group)

        return True

    def _analyzer_based_fusion_decision(self, node: OpNode, group: List[OpNode]) -> bool:
        """Make fusion decisions based on analyzer insights.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if fusion is recommended based on data characteristics
        """
        # Get dataset characteristics from analyzer
        dataset_size = self.analyzer_insights.get("dataset_size", 0)
        text_length_stats = self.analyzer_insights.get("text_length", {})
        content_ratios = self.analyzer_insights.get("content_ratios", {})

        # Decision 1: Large datasets benefit more from fusion
        if dataset_size > 100000:
            logger.debug(f"Large dataset ({dataset_size:,} samples) - favoring fusion")
            return True

        # Decision 2: High variance in text length suggests complex processing
        if text_length_stats:
            mean_length = text_length_stats.get("mean", 0)
            std_length = text_length_stats.get("std", 0)
            if mean_length > 0 and std_length / mean_length > 1.5:
                logger.debug("High text length variance - favoring fusion for complex data")
                return True

        # Decision 3: Mixed content types suggest complex processing
        multimodal_indicators = ["image_ratio", "audio_ratio", "video_ratio"]
        multimodal_count = sum(1 for indicator in multimodal_indicators if content_ratios.get(indicator, 0) > 0.1)

        if multimodal_count > 1:
            logger.debug(f"Multimodal content detected ({multimodal_count} types) - favoring fusion")
            return True

        # Decision 4: Check if operations are computationally similar
        return self._check_computational_similarity(node, group)

    def _check_computational_similarity(self, node: OpNode, group: List[OpNode]) -> bool:
        """Check if operations have similar computational characteristics.

        Args:
            node: Operation to check
            group: Group of operations

        Returns:
            True if operations are computationally similar
        """
        node_complexity = self._get_operation_complexity(node.name)
        group_complexities = [self._get_operation_complexity(op.name) for op in group]

        # Prefer grouping operations of similar complexity
        if node_complexity in group_complexities:
            return True

        # Allow mixing simple and medium operations
        if node_complexity == "simple" and all(c in ["simple", "medium"] for c in group_complexities):
            return True
        if node_complexity == "medium" and all(c in ["simple", "medium"] for c in group_complexities):
            return True

        return False

    def _get_operation_complexity(self, op_name: str) -> str:
        """Get the computational complexity of an operation.

        Args:
            op_name: Name of the operation

        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        simple_ops = {"text_length_filter", "words_num_filter", "character_repetition_filter"}
        medium_ops = {"word_repetition_filter", "special_characters_filter", "alphanumeric_filter"}
        complex_ops = {"perplexity_filter", "stopwords_filter", "flagged_words_filter"}

        if op_name in simple_ops:
            return "simple"
        elif op_name in medium_ops:
            return "medium"
        elif op_name in complex_ops:
            return "complex"
        else:
            return "medium"  # Default assumption

    def _has_dependency(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if op1 depends on op2.

        Args:
            op1: First operation
            op2: Second operation

        Returns:
            True if op1 depends on op2
        """
        # Get operation configurations
        config1 = op1.config or {}
        config2 = op2.config or {}

        # 1. Check intermediate variables
        op1_vars = set(config1.get("inter_vars", []))
        op2_vars = set(config2.get("inter_vars", []))
        if op1_vars & op2_vars:
            return True

        # 2. Check stats dependencies
        if self._check_stats_dependencies(op1, op2):
            return True

        # 3. Check model dependencies
        if self._check_model_dependencies(op1, op2):
            return True

        # 4. Check data field dependencies
        if self._check_field_dependencies(op1, op2):
            return True

        # 5. Check operation-specific dependencies
        if self._check_operation_specific_dependencies(op1, op2):
            return True

        return False

    def _check_stats_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations depend on the same stats."""
        # Get stats keys that each operation produces/consumes
        op1_stats = self._get_stats_keys(op1)
        op2_stats = self._get_stats_keys(op2)

        # If they share any stats keys, they have a dependency
        return bool(op1_stats & op2_stats)

    def _get_stats_keys(self, op: OpNode) -> set:
        """Get stats keys that an operation produces or consumes."""
        # Map operation names to their stats keys
        stats_mapping = {
            "words_num_filter": {StatsKeys.num_words},
            "text_length_filter": {StatsKeys.text_len},
            "character_repetition_filter": {StatsKeys.char_rep_ratio},
            "word_repetition_filter": {StatsKeys.word_rep_ratio},
            "average_line_length_filter": {StatsKeys.avg_line_length},
            "maximum_line_length_filter": {StatsKeys.max_line_length},
            "alphanumeric_filter": {StatsKeys.alnum_ratio, StatsKeys.alpha_token_ratio},
            "special_characters_filter": {StatsKeys.special_char_ratio},
            "perplexity_filter": {StatsKeys.perplexity},
            "stopwords_filter": {StatsKeys.stopwords_ratio},
            "flagged_words_filter": {StatsKeys.flagged_words_ratio},
            "text_entity_dependency_filter": {StatsKeys.num_dependency_edges},
            "general_field_filter": {StatsKeys.general_field_filter_condition},
        }

        return stats_mapping.get(op.name, set())

    def _check_model_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations use the same models."""
        config1 = op1.config or {}
        config2 = op2.config or {}

        # Get model keys
        op1_models = set()
        op2_models = set()

        # Check for model keys in config
        for key in ["model_key", "sp_model_key", "kl_model_key"]:
            if key in config1:
                op1_models.add(config1[key])
            if key in config2:
                op2_models.add(config2[key])

        # If they share any models, they have a dependency
        return bool(op1_models & op2_models)

    def _check_field_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check if operations process the same data fields."""
        config1 = op1.config or {}
        config2 = op2.config or {}

        # Get field keys
        op1_fields = set()
        op2_fields = set()

        # Check for field keys in config
        for key in ["text_key", "image_key", "audio_key", "video_key"]:
            if key in config1:
                op1_fields.add(config1[key])
            if key in config2:
                op2_fields.add(config2[key])

        # If they share any fields, they might have a dependency
        # (This is a conservative check - some operations can share fields safely)
        shared_fields = op1_fields & op2_fields

        # Only consider it a dependency if both operations are text processors
        # and they share text_key (indicating they process the same text)
        if shared_fields and "text_key" in shared_fields:
            return True

        return False

    def _check_operation_specific_dependencies(self, op1: OpNode, op2: OpNode) -> bool:
        """Check operation-specific dependencies."""
        # Some operations have specific dependencies that can't be generalized

        # Example: Operations that modify the same data structure
        # This is a placeholder for future operation-specific checks
        return False
