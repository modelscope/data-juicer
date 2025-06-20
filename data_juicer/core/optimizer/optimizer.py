from typing import List, Optional

from loguru import logger

from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy
from data_juicer.core.optimizer.mapper_fusion_strategy import MapperFusionStrategy
from data_juicer.core.optimizer.strategy import OptimizationStrategy
from data_juicer.core.pipeline_ast import PipelineAST


class PipelineOptimizer:
    """Main optimizer class that manages multiple optimization strategies."""

    def __init__(self, strategies: Optional[List[OptimizationStrategy]] = None):
        """Initialize the optimizer with a list of strategies.

        Args:
            strategies: List of optimization strategies to apply. If None,
                       default strategies will be used.
        """
        self.strategies = strategies or [MapperFusionStrategy(), FilterFusionStrategy()]

    def add_strategy(self, strategy: OptimizationStrategy) -> None:
        """Add a new optimization strategy.

        Args:
            strategy: The optimization strategy to add.
        """
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_name: str) -> None:
        """Remove an optimization strategy by name.

        Args:
            strategy_name: Name of the strategy to remove.
        """
        self.strategies = [s for s in self.strategies if s.name != strategy_name]

    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply all optimization strategies to the AST.

        Args:
            ast: The pipeline AST to optimize.

        Returns:
            The optimized AST.
        """
        if not ast.root:
            logger.warning("Empty pipeline, nothing to optimize")
            return ast

        logger.info(f"Starting pipeline optimization with {len(self.strategies)} strategies")

        # Apply each strategy in sequence
        for strategy in self.strategies:
            logger.info(f"Applying optimization strategy: {strategy.name}")
            ast = strategy.optimize(ast)

        logger.info("Pipeline optimization completed")
        return ast

    def get_strategy(self, strategy_name: str) -> Optional[OptimizationStrategy]:
        """Get a strategy by name.

        Args:
            strategy_name: Name of the strategy to get.

        Returns:
            The strategy if found, None otherwise.
        """
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                return strategy
        return None

    def get_strategy_names(self) -> List[str]:
        """Get names of all registered strategies.

        Returns:
            List of strategy names.
        """
        return [strategy.name for strategy in self.strategies]

    def clear_strategies(self) -> None:
        """Remove all optimization strategies."""
        self.strategies = []
