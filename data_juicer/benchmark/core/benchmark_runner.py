#!/usr/bin/env python3
"""
Main benchmark runner for executing performance tests.
"""

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..utils.config_manager import ConfigManager
from .metrics_collector import BenchmarkMetrics, MetricsCollector


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    dataset_path: str
    config_path: str
    output_dir: str
    iterations: int = 3
    warmup_runs: int = 1
    timeout_seconds: int = 3600
    strategy_name: str = "baseline"
    strategy_config: Dict[str, Any] = None


class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.config_manager = ConfigManager()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_benchmark(self) -> BenchmarkMetrics:
        """Run a single benchmark iteration."""
        logger.info(f"Starting benchmark: {self.config.strategy_name}")

        # Start metrics collection
        self.metrics_collector.start_monitoring()

        try:
            # Prepare configuration
            config_file = self._prepare_config()

            # Run the benchmark
            start_time = time.time()
            result = self._execute_benchmark(config_file)
            end_time = time.time()

            # Stop metrics collection
            metrics = self.metrics_collector.stop_monitoring()

            # Enhance metrics with benchmark-specific data
            if metrics:
                metrics.total_wall_time = end_time - start_time
                metrics.strategy_name = self.config.strategy_name
                metrics.config_hash = self._get_config_hash()

                # Add benchmark-specific metrics
                if result:
                    metrics.samples_processed = result.get("samples_processed", 0)
                    metrics.samples_retained = result.get("samples_retained", 0)
                    metrics.samples_per_second = (
                        metrics.samples_processed / metrics.total_wall_time if metrics.total_wall_time > 0 else 0
                    )
                    metrics.data_retention_rate = (
                        metrics.samples_retained / metrics.samples_processed if metrics.samples_processed > 0 else 0
                    )

            return metrics

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            # Still try to get partial metrics
            metrics = self.metrics_collector.stop_monitoring()
            if metrics:
                metrics.strategy_name = self.config.strategy_name
                metrics.config_hash = self._get_config_hash()
            return metrics

    def run_benchmark_suite(self) -> List[BenchmarkMetrics]:
        """Run multiple iterations of the benchmark."""
        logger.info(f"Running benchmark suite: {self.config.iterations} iterations")

        all_metrics = []

        # Warmup runs (not counted in results)
        for i in range(self.config.warmup_runs):
            logger.info(f"Warmup run {i+1}/{self.config.warmup_runs}")
            self.run_benchmark()

        # Actual benchmark runs
        for i in range(self.config.iterations):
            logger.info(f"Benchmark iteration {i+1}/{self.config.iterations}")
            metrics = self.run_benchmark()
            if metrics:
                all_metrics.append(metrics)

        return all_metrics

    def _prepare_config(self) -> str:
        """Prepare configuration file for the benchmark."""
        # Load base configuration
        base_config = self.config_manager.load_config(self.config.config_path)

        # Apply strategy-specific modifications
        if self.config.strategy_config:
            base_config = self.config_manager.apply_strategy_config(base_config, self.config.strategy_config)

        # Save modified configuration
        config_output_path = os.path.join(self.config.output_dir, f"config_{self.config.strategy_name}.yaml")
        self.config_manager.save_config(base_config, config_output_path)

        return config_output_path

    def _execute_benchmark(self, config_file: str) -> Optional[Dict[str, Any]]:
        """Execute the actual benchmark using data-juicer."""
        try:
            # Build command
            cmd = [
                "python",
                "-m",
                "data_juicer.tools.process_data",
                "--config",
                config_file,
                "--dataset_path",
                self.config.dataset_path,
                "--output_dir",
                os.path.join(self.config.output_dir, "output"),
            ]

            logger.debug(f"Executing command: {' '.join(cmd)}")

            # Run the benchmark
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds, cwd=os.getcwd()
            )

            if result.returncode != 0:
                logger.error(f"Benchmark execution failed: {result.stderr}")
                return None

            # Parse output for metrics
            return self._parse_benchmark_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {self.config.timeout_seconds} seconds")
            return None
        except Exception as e:
            logger.error(f"Error executing benchmark: {e}")
            return None

    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse benchmark output to extract metrics."""
        # This is a simplified parser - in practice, you'd want more robust parsing
        metrics = {}

        # Look for common patterns in data-juicer output
        lines = output.split("\n")
        for line in lines:
            if "processed" in line.lower() and "samples" in line.lower():
                # Try to extract sample count
                try:
                    # This is a simplified extraction - would need more robust parsing
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i > 0:
                            metrics["samples_processed"] = int(part)
                            break
                except Exception as e:
                    logger.error(f"Error parsing benchmark output: {e}")
                    pass

        return metrics

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration."""
        config_str = f"{self.config.dataset_path}_{self.config.config_path}_{self.config.strategy_name}"
        if self.config.strategy_config:
            config_str += str(sorted(self.config.strategy_config.items()))

        return hashlib.md5(config_str.encode()).hexdigest()[:8]
