#!/usr/bin/env python3
"""
Concrete strategy implementations for the benchmark framework.
"""

from typing import Any, Dict, List

from .strategy_library import OptimizationStrategy, StrategyType


class CoreOptimizerStrategy(OptimizationStrategy):
    """Strategy that configures the core optimizer to enable/disable specific strategies."""

    def __init__(self, name: str, description: str, enabled_strategies: List[str]):
        super().__init__(name, description)
        self.strategy_type = StrategyType.FUSION  # Core optimizer is primarily fusion-based
        self.enabled_strategies = enabled_strategies

    def apply_to_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply core optimizer configuration."""
        # For now, return the config unchanged since we can't add custom keys
        # The actual optimizer integration would need to be implemented at the benchmark runner level
        # TODO: Implement proper core optimizer integration in benchmark runner
        return config.copy()

    def get_expected_impact(self) -> Dict[str, str]:
        """Get expected impact description."""
        return {
            "performance": "Improved performance through core optimizer strategies",
            "memory": "Optimized memory usage through operation fusion",
            "complexity": "Moderate configuration complexity",
        }


class BaselineStrategy(OptimizationStrategy):
    """Baseline strategy with no optimizations."""

    def __init__(self, name: str = "baseline"):
        super().__init__(name, "Baseline configuration with no optimizations")
        self.strategy_type = StrategyType.ALGORITHM

    def apply_to_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply baseline configuration (no changes)."""
        # Return config unchanged for baseline
        return config.copy()

    def get_expected_impact(self) -> Dict[str, str]:
        """Get expected impact description."""
        return {
            "performance": "Baseline performance",
            "memory": "Standard memory usage",
            "complexity": "Minimal configuration complexity",
        }
