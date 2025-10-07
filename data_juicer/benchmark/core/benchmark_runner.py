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
    sample_ratio: float = 1.0
    sample_method: str = "random"


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
            # Check if this is a core optimizer strategy
            if self.config.strategy_config.get("enable_core_optimizer"):
                # For core optimizer strategies, we need to modify the pipeline
                # Add a note to the config that core optimizer should be enabled
                base_config["_benchmark_optimizer_enabled"] = True
                base_config["_benchmark_optimizer_strategies"] = self.config.strategy_config.get(
                    "optimizer_strategies", []
                )
            else:
                # For regular config strategies, apply them directly
                base_config = self.config_manager.apply_strategy_config(base_config, self.config.strategy_config)

        # Apply sampling if needed
        if self.config.sample_ratio < 1.0:
            base_config = self._apply_sampling_config(base_config)

        # Save modified configuration
        config_output_path = os.path.join(self.config.output_dir, f"config_{self.config.strategy_name}.yaml")
        self.config_manager.save_config(base_config, config_output_path)

        return config_output_path

    def _apply_sampling_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sampling configuration to the config."""
        # If sampling is enabled, we'll need to create a sampled dataset
        if self.config.sample_ratio < 1.0:
            sampled_dataset_path = self._create_sampled_dataset()
            config["dataset_path"] = sampled_dataset_path

        return config

    def _create_sampled_dataset(self) -> str:
        """Create a sampled version of the dataset."""
        import random

        # Read the original dataset
        with open(self.config.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_samples = len(lines)
        sample_size = int(total_samples * self.config.sample_ratio)

        # Sample the data based on method
        if self.config.sample_method == "random":
            sampled_lines = random.sample(lines, sample_size)
        elif self.config.sample_method == "first":
            sampled_lines = lines[:sample_size]
        elif self.config.sample_method == "last":
            sampled_lines = lines[-sample_size:]
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sample_method}")

        # Create sampled dataset file
        sampled_path = os.path.join(
            self.config.output_dir, f"sampled_dataset_{self.config.sample_ratio}_{self.config.sample_method}.jsonl"
        )

        with open(sampled_path, "w", encoding="utf-8") as f:
            f.writelines(sampled_lines)

        logger.info(f"Created sampled dataset: {sampled_path} ({sample_size}/{total_samples} samples)")
        return sampled_path

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
                "--export_path",
                os.path.join(self.config.output_dir, "output.jsonl"),
            ]

            logger.debug(f"Executing command: {' '.join(cmd)}")

            # Run the benchmark
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds, cwd=os.getcwd()
            )

            if result.returncode != 0:
                logger.error(f"Benchmark execution failed: {result.stderr}")
                return None

            # Log the subprocess output for debugging
            logger.info("=== Subprocess STDOUT ===")
            logger.info(result.stdout)
            logger.info("=== Subprocess STDERR ===")
            logger.info(result.stderr)
            logger.info("=== End Subprocess Output ===")

            # Parse output for metrics (data-juicer logs to stderr, not stdout)
            return self._parse_benchmark_output(result.stderr)

        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {self.config.timeout_seconds} seconds")
            return None
        except Exception as e:
            logger.error(f"Error executing benchmark: {e}")
            return None

    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse benchmark output to extract metrics."""
        metrics = {}

        # Look for patterns in data-juicer output
        lines = output.split("\n")

        # Track sample counts through processing steps
        initial_samples = None
        final_samples = None

        for line in lines:
            # Look for initial sample count (after filtering empty text)
            if "samples left after filtering empty text" in line:
                try:
                    # Extract number from line like "10000 samples left after filtering empty text"
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            initial_samples = int(part)
                            break
                except Exception as e:
                    logger.error(f"Error parsing initial samples: {e}")

            # Look for final sample count (last "Left X samples" in processing)
            if "Left" in line and "samples" in line and "Done in" in line:
                try:
                    # Extract number from line like "Left 1000 samples"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Left" and i + 1 < len(parts):
                            if parts[i + 1].isdigit():
                                final_samples = int(parts[i + 1])
                                break
                except Exception as e:
                    logger.error(f"Error parsing final samples: {e}")

        # Set metrics based on what we found
        if initial_samples is not None:
            metrics["samples_processed"] = initial_samples
        if final_samples is not None:
            metrics["samples_retained"] = final_samples

        logger.info(f"=== Metrics Parsing Results ===")
        logger.info(f"Initial samples found: {initial_samples}")
        logger.info(f"Final samples found: {final_samples}")
        logger.info(f"Parsed metrics: {metrics}")
        logger.info(f"=== End Metrics Parsing ===")
        return metrics

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration."""
        config_str = f"{self.config.dataset_path}_{self.config.config_path}_{self.config.strategy_name}"
        if self.config.strategy_config:
            config_str += str(sorted(self.config.strategy_config.items()))

        return hashlib.md5(config_str.encode()).hexdigest()[:8]
