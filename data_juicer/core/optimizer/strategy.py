from abc import ABC, abstractmethod
from typing import List

from data_juicer.core.pipeline_ast import OpNode, PipelineAST


class OptimizationStrategy(ABC):
    """Base class for pipeline optimization strategies."""

    def __init__(self, name: str):
        """Initialize the optimization strategy.

        Args:
            name: Name of the optimization strategy
        """
        self.name = name

    @abstractmethod
    def optimize(self, ast: PipelineAST) -> PipelineAST:
        """Apply the optimization strategy to the pipeline AST.

        Args:
            ast: The pipeline AST to optimize

        Returns:
            Optimized pipeline AST
        """
        pass

    def _get_operation_chain(self, node: OpNode) -> List[OpNode]:
        """Get the linear chain of operations from a node.

        Args:
            node: The node to start from

        Returns:
            List of operations in the chain
        """
        chain = []
        current = node
        while current.children:
            current = current.children[0]
            chain.append(current)
        return chain

    def _rebuild_chain(self, root: OpNode, chain: List[OpNode]) -> None:
        """Rebuild the operation chain from a list of nodes.

        Args:
            root: The root node
            chain: List of operations to chain
        """
        current = root
        for node in chain:
            current.children = [node]
            node.parent = current
            current = node
