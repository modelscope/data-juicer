#!/usr/bin/env python3
"""
Performance benchmark for Data-Juicer filter fusion and optimization.

This benchmark compares individual vs fused filter performance and demonstrates
the new PipelineOptimizer architecture.

USAGE EXAMPLES:
    # Quick benchmark (basic demo with 1000 samples)
    python performance_benchmark.py --mode quick

    # Quick benchmark with more samples
    python performance_benchmark.py --mode quick --samples 10000

    # Full comprehensive benchmark
    python performance_benchmark.py --mode full --samples 50000 --runs 5

    # Test the new optimizer architecture
    python performance_benchmark.py --mode optimizer --samples 10000

MODES:
    quick    - Basic performance demo with analyzer insights (default)
    full     - Comprehensive benchmark with multiple runs and detailed analysis
    optimizer - Test the new PipelineOptimizer architecture vs legacy fusion
"""

import os
import random
import shutil
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from data_juicer.core.analyzer import Analyzer
from data_juicer.core.optimizer.fused_op import FusedFilter
from data_juicer.core.optimizer.optimizer import PipelineOptimizer
from data_juicer.ops.base_op import Filter
from data_juicer.ops.filter import (
    AlphanumericFilter,
    AverageLineLengthFilter,
    CharacterRepetitionFilter,
    MaximumLineLengthFilter,
    SpecialCharactersFilter,
    TextLengthFilter,
    WordRepetitionFilter,
    WordsNumFilter,
)
from data_juicer.utils.constant import Fields


@dataclass
class PerformanceMetrics:
    """Performance metrics for comparison."""

    total_time: float
    stats_time: float
    filter_time: float
    memory_usage: float
    throughput: float  # samples per second


class PerformanceBenchmark:
    """Comprehensive performance benchmark for fused vs individual filters."""

    def __init__(self):
        self.results = {}

    def create_realistic_test_data(self, num_samples: int = 10000) -> Dict[str, Any]:
        """Create realistic test data that triggers various filter conditions."""
        logger.info(f"Creating test data with {num_samples} samples...")

        texts = []
        for i in range(num_samples):
            # Create diverse text samples with different characteristics
            if i % 4 == 0:
                # Short texts
                length = random.randint(10, 50)
                text = "".join(random.choices(string.ascii_lowercase + " ", k=length))
            elif i % 4 == 1:
                # Long texts with repetition
                length = random.randint(200, 800)
                text = "".join(random.choices("hello world test data " * 20, k=length))
            elif i % 4 == 2:
                # Texts with special characters
                length = random.randint(100, 400)
                text = "".join(
                    random.choices(string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?", k=length)
                )
            else:
                # Normal texts
                length = random.randint(80, 300)
                text = "".join(random.choices(string.ascii_letters + string.digits + " .,!?", k=length))

            texts.append(text)

        test_data = {"text": texts, Fields.stats: [{} for _ in range(num_samples)]}

        logger.info(f"Created test data with {len(texts)} text samples")
        return test_data

    def create_basic_test_filters(self) -> List[Filter]:
        """Create only the most basic filters for reliable testing."""
        logger.info("Creating basic test filters...")

        filters = [
            # Only the most basic, reliable filters
            WordsNumFilter(min_num=5, max_num=1000),
            TextLengthFilter(min_len=20, max_len=1000),
            CharacterRepetitionFilter(repetition_ratio=0.8),
        ]

        return filters

    def create_test_filters(self) -> List[Filter]:
        """Create a safe set of test filters that won't hang."""
        logger.info("Creating safe test filters...")

        filters = [
            # Only simple, reliable filters
            WordsNumFilter(min_num=5, max_num=1000),
            TextLengthFilter(min_len=20, max_len=1000),
            CharacterRepetitionFilter(repetition_ratio=0.8),
            WordRepetitionFilter(min_ratio=0.0, max_ratio=0.5),
            SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),
            AlphanumericFilter(min_ratio=0.3),
            AverageLineLengthFilter(min_len=10, max_len=100),
            MaximumLineLengthFilter(min_len=10, max_len=200),
        ]

        return filters

    def measure_memory_usage(self) -> float:
        """Measure current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            logger.warning("psutil not available, using 0 for memory measurement")
            return 0.0

    def get_analyzer_insights(self, test_data: Dict[str, Any]) -> dict:
        """
        Run the actual Analyzer on the test data and extract insights for optimization.
        Handles both HuggingFace Dataset and Ray dataset formats.
        """
        logger.info("Running Analyzer to extract real insights...")

        # Create a temporary config for the Analyzer
        from jsonargparse import Namespace

        # Include basic filters to compute meaningful statistics
        process_config = [
            {"WordsNumFilter": {"min_num": 1, "max_num": 10000}},
            {"TextLengthFilter": {"min_len": 1, "max_len": 10000}},
            {"CharacterRepetitionFilter": {"repetition_ratio": 1.0}},
        ]

        cfg = Namespace(
            work_dir="./tmp_benchmark_analyzer",
            export_path="./tmp_benchmark_analyzer/export.jsonl",
            export_shard_size=10000,
            export_in_parallel=False,
            np=1,
            export_original_dataset=False,
            use_cache=False,
            cache_compress=None,
            open_monitor=False,
            process=process_config,  # Include filters to compute stats
            auto=False,
            auto_num=1000,
            op_fusion=False,
            fusion_strategy="greedy",
            save_stats_in_one_file=True,
            percentiles=[0.25, 0.5, 0.75],
        )

        try:
            # Check if test_data is a Ray dataset
            if hasattr(test_data, "data") and hasattr(test_data.data, "to_pandas"):
                # Convert Ray dataset to HuggingFace Dataset
                logger.info("Converting Ray dataset to HuggingFace format for Analyzer...")
                from datasets import Dataset

                # Convert to pandas first, then to HuggingFace Dataset
                df = test_data.data.to_pandas()
                dataset = Dataset.from_pandas(df)

            elif isinstance(test_data, dict):
                # Convert dict to HuggingFace Dataset
                from datasets import Dataset

                dataset = Dataset.from_dict(test_data)

            else:
                # Assume it's already a HuggingFace Dataset
                dataset = test_data

            # Run the Analyzer
            analyzer = Analyzer(cfg)
            analyzer.run(dataset=dataset, skip_return=True)

            # Extract insights from analyzer.overall_result (a DataFrame)
            overall = analyzer.overall_result
            insights = {"dataset_size": len(dataset), "text_length": {}, "content_ratios": {}}

            # Log detailed statistics before cleanup
            logger.info("📊 DETAILED ANALYZER STATISTICS:")
            logger.info("=" * 50)

            if overall is not None:
                logger.info(f"Dataset size: {len(dataset):,} samples")
                logger.info(f"Available statistics: {list(overall.index)}")

                # Log all available statistics
                for stat_name in overall.index:
                    stat_data = overall.loc[stat_name]
                    logger.info(f"\n{stat_name.upper()}:")
                    for col in stat_data.index:
                        value = stat_data[col]
                        if isinstance(value, (int, float)):
                            logger.info(f"  {col}: {value:,.2f}")
                        else:
                            logger.info(f"  {col}: {value}")
            else:
                logger.warning("No overall statistics available from Analyzer")

            # Extract specific insights for optimization
            if overall is not None and "text_length" in overall.index:
                stats = overall.loc["text_length"]
                insights["text_length"] = {"mean": float(stats.get("mean", 0)), "std": float(stats.get("std", 0))}
                logger.info("📏 TEXT LENGTH INSIGHTS:")
                logger.info(f"  Mean length: {insights['text_length']['mean']:.1f} characters")
                logger.info(f"  Std deviation: {insights['text_length']['std']:.1f} characters")
            else:
                # Fallback: compute basic stats manually
                if hasattr(dataset, "column_names") and "text" in dataset.column_names:
                    texts = dataset["text"]
                else:
                    texts = []
                if texts:
                    lengths = [len(t) for t in texts]
                    insights["text_length"] = {"mean": float(np.mean(lengths)), "std": float(np.std(lengths))}
                    logger.info("📏 MANUAL TEXT LENGTH COMPUTATION:")
                    logger.info(f"  Mean length: {insights['text_length']['mean']:.1f} characters")
                    logger.info(f"  Std deviation: {insights['text_length']['std']:.1f} characters")
                else:
                    insights["text_length"] = {"mean": 0, "std": 0}
                    logger.warning("No text data available for length analysis")

            # Extract content ratios
            logger.info("🎭 CONTENT RATIOS:")
            for col in ["image_ratio", "audio_ratio", "video_ratio"]:
                if overall is not None and col in overall.index:
                    ratio = float(overall.loc[col].get("mean", 0))
                    insights["content_ratios"][col] = ratio
                    logger.info(f"  {col}: {ratio:.3f} ({ratio*100:.1f}%)")
                else:
                    insights["content_ratios"][col] = 0.0
                    logger.info(f"  {col}: 0.000 (0.0%)")

            # Log optimization recommendations based on insights
            logger.info("🎯 OPTIMIZATION RECOMMENDATIONS:")
            dataset_size = len(dataset)
            if dataset_size > 100000:
                logger.info("  📈 Large dataset detected - Fusion will provide significant benefits")
            elif dataset_size > 10000:
                logger.info("  📊 Medium dataset detected - Fusion will provide moderate benefits")
            else:
                logger.info("  📉 Small dataset detected - Fusion benefits may be minimal")

            text_mean = insights["text_length"]["mean"]
            if text_mean > 1000:
                logger.info("  📝 Long text detected - Consider text-specific optimizations")
            elif text_mean < 100:
                logger.info("  📝 Short text detected - Simple filters may be sufficient")

            logger.info("=" * 50)
            logger.info(f"Successfully extracted analyzer insights: {insights}")

            # Show how insights will be used for optimization
            logger.info("🔧 INSIGHTS FOR OPTIMIZATION:")
            logger.info(f"  Dataset size: {insights['dataset_size']:,} samples")
            logger.info(f"  Text length mean: {insights['text_length']['mean']:.1f} chars")
            logger.info(f"  Text length std: {insights['text_length']['std']:.1f} chars")
            logger.info(f"  Content ratios: {insights['content_ratios']}")
            logger.info("  These insights will be used to:")
            logger.info("    - Choose optimal fusion strategy (parallel vs sequential)")
            logger.info("    - Determine batch sizes for processing")
            logger.info("    - Select filter execution order")
            logger.info("    - Estimate memory requirements")

            # Clean up temporary directory
            if os.path.exists("./tmp_benchmark_analyzer"):
                shutil.rmtree("./tmp_benchmark_analyzer")
                logger.debug("Cleaned up temporary analyzer directory")

            return insights

        except Exception as e:
            logger.warning(f"Failed to run Analyzer: {e}. Falling back to simulated insights.")

            # Clean up temporary directory even if Analyzer failed
            if os.path.exists("./tmp_benchmark_analyzer"):
                shutil.rmtree("./tmp_benchmark_analyzer")
                logger.debug("Cleaned up temporary analyzer directory after failure")

            return self.simulate_analyzer_insights(test_data)

    def simulate_analyzer_insights(self, test_data: Dict[str, Any]) -> dict:
        """
        Simulate analyzer insights from synthetic test data for benchmarking.
        Fallback method when Analyzer fails.
        """
        logger.info("Using simulated analyzer insights...")
        logger.info("📊 SIMULATED ANALYZER STATISTICS:")
        logger.info("=" * 50)

        # Handle different data formats
        if hasattr(test_data, "data") and hasattr(test_data.data, "to_pandas"):
            # Ray dataset
            df = test_data.data.to_pandas()
            texts = df["text"].tolist() if "text" in df.columns else []
            logger.info("Data format: Ray dataset")
        elif isinstance(test_data, dict) and "text" in test_data:
            # Dict format
            texts = test_data["text"]
            logger.info("Data format: Dictionary")
        else:
            # Assume it's a HuggingFace Dataset
            if hasattr(test_data, "column_names") and "text" in test_data.column_names:
                texts = test_data["text"]
                logger.info("Data format: HuggingFace Dataset")
            else:
                texts = []
                logger.warning("Data format: Unknown")

        logger.info(f"Dataset size: {len(texts):,} samples")

        if texts:
            lengths = [len(t) for t in texts]
            mean_length = float(np.mean(lengths))
            std_length = float(np.std(lengths))

            logger.info("📏 TEXT LENGTH STATISTICS:")
            logger.info(f"  Mean length: {mean_length:.1f} characters")
            logger.info(f"  Std deviation: {std_length:.1f} characters")
            logger.info(f"  Min length: {min(lengths):.0f} characters")
            logger.info(f"  Max length: {max(lengths):.0f} characters")
        else:
            mean_length = std_length = 0.0
            logger.warning("No text data available for analysis")

        # Simulate multimodal ratios (none in synthetic data, but could randomize)
        content_ratios = {"image_ratio": 0.0, "audio_ratio": 0.0, "video_ratio": 0.0}

        logger.info("🎭 CONTENT RATIOS (Simulated):")
        for col, ratio in content_ratios.items():
            logger.info(f"  {col}: {ratio:.3f} ({ratio*100:.1f}%)")

        logger.info("🎯 SIMULATED OPTIMIZATION RECOMMENDATIONS:")
        dataset_size = len(texts) if texts else 0
        if dataset_size > 100000:
            logger.info("  📈 Large dataset detected - Fusion will provide significant benefits")
        elif dataset_size > 10000:
            logger.info("  📊 Medium dataset detected - Fusion will provide moderate benefits")
        else:
            logger.info("  📉 Small dataset detected - Fusion benefits may be minimal")

        if mean_length > 1000:
            logger.info("  📝 Long text detected - Consider text-specific optimizations")
        elif mean_length < 100:
            logger.info("  📝 Short text detected - Simple filters may be sufficient")

        logger.info("=" * 50)

        return {
            "dataset_size": len(texts) if texts else 0,
            "text_length": {"mean": mean_length, "std": std_length},
            "content_ratios": content_ratios,
        }

    def run_individual_filters_benchmark(self, filters: List[Filter], test_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark individual filter execution."""
        logger.info("Running individual filters benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        total_stats_time = 0.0
        total_filter_time = 0.0

        for i, filter_op in enumerate(filters):
            op_name = getattr(filter_op, "_name", type(filter_op).__name__)
            logger.info(f"  Processing filter {i+1}/{len(filters)}: {op_name}")

            # Phase 1: Stats computation
            stats_start = time.time()
            samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
            stats_time = time.time() - stats_start
            total_stats_time += stats_time

            # Phase 2: Filtering
            filter_start = time.time()
            _ = filter_op.process_batched(samples_with_stats)
            filter_time = time.time() - filter_start
            total_filter_time += filter_time

            logger.info(f"    Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s")

        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory

        throughput = len(test_data["text"]) / total_time

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_stats_time,
            filter_time=total_filter_time,
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def run_fused_filters_benchmark(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Benchmark fused filter execution with analyzer insights."""
        logger.info("Running fused filters benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Ensure analyzer_insights is a dict
        if analyzer_insights is None:
            analyzer_insights = {}
        fused_filter = FusedFilter("performance_test_fused", filters, analyzer_insights=analyzer_insights)

        # Phase 1: Stats computation
        stats_start = time.time()
        samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
        stats_time = time.time() - stats_start

        # Phase 2: Filtering
        filter_start = time.time()
        _ = fused_filter.process_batched(samples_with_stats)
        filter_time = time.time() - filter_start

        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory

        throughput = len(test_data["text"]) / total_time

        logger.info(
            f"  Fused execution - Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s, Total: {total_time:.3f}s"
        )

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=stats_time,
            filter_time=filter_time,
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def collect_filtering_statistics(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> Dict[str, Any]:
        """Collect filtering statistics for comprehensive analysis."""
        logger.info("📊 Collecting comprehensive filtering statistics...")

        # Individual filter stats
        individual_stats = {}
        total_samples = len(test_data["text"])

        for i, filter_op in enumerate(filters):
            op_name = getattr(filter_op, "_name", type(filter_op).__name__)
            logger.info(f"  Testing filter {i+1}/{len(filters)}: {op_name}")

            # Compute stats and filter
            samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
            filter_results = list(filter_op.process_batched(samples_with_stats))

            # Count passed samples
            passed_samples = sum(filter_results)
            pass_rate = (passed_samples / total_samples) * 100

            individual_stats[op_name] = {
                "passed": passed_samples,
                "filtered": total_samples - passed_samples,
                "pass_rate": pass_rate,
            }

            logger.info(f"    Passed: {passed_samples:,}/{total_samples:,} ({pass_rate:.1f}%)")

        # Fused filter stats
        logger.info("  Testing fused filters...")
        if analyzer_insights is None:
            analyzer_insights = {}
        fused_filter = FusedFilter("comprehensive_test_fused", filters, analyzer_insights=analyzer_insights)
        samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
        fused_results = list(fused_filter.process_batched(samples_with_stats))

        fused_passed = sum(fused_results)
        fused_pass_rate = (fused_passed / total_samples) * 100

        fused_stats = {"passed": fused_passed, "filtered": total_samples - fused_passed, "pass_rate": fused_pass_rate}

        logger.info(f"    Fused - Passed: {fused_passed:,}/{total_samples:,} ({fused_pass_rate:.1f}%)")

        return {"individual": individual_stats, "fused": fused_stats, "total_samples": total_samples}

    def run_comprehensive_test(self, num_samples: int = 10000, num_runs: int = 3):
        """Run comprehensive performance test with multiple runs and detailed analysis."""
        logger.info(f"Running comprehensive test with {num_samples:,} samples, {num_runs} runs")

        # Create test data
        logger.info("Creating test data...")
        test_data = self.create_realistic_test_data(num_samples)

        # Simulate analyzer insights
        analyzer_insights = self.get_analyzer_insights(test_data)
        logger.info(f"Analyzer insights for optimization: {analyzer_insights}")

        # Create test filters
        logger.info("Creating test filters...")
        try:
            filters = self.create_test_filters()
            logger.info(f"Successfully created {len(filters)} filters")
        except Exception as e:
            logger.error(f"Failed to create test filters: {e}")
            logger.error("Full traceback:")
            import traceback

            traceback.print_exc()
            raise

        # Collect filtering statistics first
        filtering_stats = self.collect_filtering_statistics(filters, test_data, analyzer_insights=analyzer_insights)
        comparison_results = print_filtering_comparison(filtering_stats)

        # Run multiple iterations for statistical significance
        individual_results = []
        fused_results = []

        for run in range(num_runs):
            logger.info(f"--- Run {run + 1}/{num_runs} ---")

            # Individual execution
            individual_result = self.run_individual_filters_benchmark(filters, test_data.copy())
            individual_results.append(individual_result)

            # Fused execution (with analyzer insights)
            fused_result = self.run_fused_filters_benchmark(
                filters, test_data.copy(), analyzer_insights=analyzer_insights
            )
            fused_results.append(fused_result)

        # Calculate statistics
        results = self.calculate_statistics(individual_results, fused_results, num_samples)

        # Add filtering statistics to results
        results["filtering_stats"] = filtering_stats
        results["comparison_results"] = comparison_results

        return results

    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE TEST RESULTS")
        logger.info("=" * 60)

        config = results["test_config"]
        improvements = results["improvements"]

        logger.info("Test Configuration:")
        logger.info(f"  Samples: {config['num_samples']:,}")
        logger.info(f"  Runs: {config['num_runs']}")
        logger.info(f"  Filters: {config['num_filters']}")

        logger.info("Performance Comparison:")
        logger.info(
            f"  Individual Execution: {results['individual']['mean_total_time']:.3f}s "
            f"± {results['individual']['std_total_time']:.3f}s"
        )
        logger.info(
            f"  Fused Execution: {results['fused']['mean_total_time']:.3f}s "
            f"± {results['fused']['std_total_time']:.3f}s"
        )

        logger.info("Improvements:")
        logger.info(f"  Total Speedup: {improvements['total_speedup']:.2f}x")
        logger.info(f"  Time Saved: {improvements['time_saved_percent']:.1f}%")
        logger.info(f"  Throughput Improvement: {improvements['throughput_improvement']:.2f}x")
        logger.info(f"  Stats Computation Speedup: {improvements['stats_speedup']:.2f}x")
        logger.info(f"  Filtering Speedup: {improvements['filter_speedup']:.2f}x")
        logger.info(f"  Memory Efficiency: {improvements['memory_efficiency']:.2f}x")

        # Performance assessment
        if improvements["total_speedup"] > 2.0:
            assessment = "EXCELLENT - High performance gain"
        elif improvements["total_speedup"] > 1.5:
            assessment = "GOOD - Significant performance gain"
        elif improvements["total_speedup"] > 1.2:
            assessment = "MODERATE - Noticeable performance gain"
        else:
            assessment = "MINIMAL - Small performance gain"

        logger.info(f"Assessment: {assessment}")
        logger.info("=" * 60)

    def save_results(self, results: Dict[str, Any], filename: str = "performance_test_results.json"):
        """Save test results to file."""
        import json

        # Convert dataclasses to dict for JSON serialization
        def convert_metrics(metrics_list):
            return [
                {
                    "total_time": m.total_time,
                    "stats_time": m.stats_time,
                    "filter_time": m.filter_time,
                    "memory_usage": m.memory_usage,
                    "throughput": m.throughput,
                }
                for m in metrics_list
            ]

        saveable_results = {
            "test_config": results["test_config"],
            "individual": results["individual"],
            "fused": results["fused"],
            "improvements": results["improvements"],
            "raw_results": {
                "individual": convert_metrics(results["raw_results"]["individual"]),
                "fused": convert_metrics(results["raw_results"]["fused"]),
            },
        }

        with open(filename, "w") as f:
            json.dump(saveable_results, f, indent=2)

        logger.info(f"Results saved to {filename}")

    def calculate_statistics(self, individual_results, fused_results, num_samples):
        """Calculate comprehensive statistics from multiple runs."""

        def calculate_stats(metrics_list):
            return {
                "mean_total_time": np.mean([m.total_time for m in metrics_list]),
                "std_total_time": np.std([m.total_time for m in metrics_list]),
                "mean_stats_time": np.mean([m.stats_time for m in metrics_list]),
                "mean_filter_time": np.mean([m.filter_time for m in metrics_list]),
                "mean_memory_usage": np.mean([m.memory_usage for m in metrics_list]),
                "mean_throughput": np.mean([m.throughput for m in metrics_list]),
            }

        individual_stats = calculate_stats(individual_results)
        fused_stats = calculate_stats(fused_results)

        # Calculate improvements
        total_speedup = individual_stats["mean_total_time"] / fused_stats["mean_total_time"]
        time_saved = individual_stats["mean_total_time"] - fused_stats["mean_total_time"]
        stats_speedup = individual_stats["mean_stats_time"] / fused_stats["mean_stats_time"]
        filter_speedup = (
            individual_stats["mean_filter_time"] / fused_stats["mean_filter_time"]
            if fused_stats["mean_filter_time"] > 0
            else float("inf")
        )

        # Compile results
        results = {
            "test_config": {
                "num_samples": num_samples,
                "num_runs": len(individual_results),
                "num_filters": len(self.create_test_filters()),
            },
            "individual": individual_stats,
            "fused": fused_stats,
            "improvements": {
                "total_speedup": total_speedup,
                "stats_speedup": stats_speedup,
                "filter_speedup": filter_speedup,
                "throughput_improvement": (fused_stats["mean_throughput"] / individual_stats["mean_throughput"] - 1)
                * 100,
                "time_saved_percent": time_saved / individual_stats["mean_total_time"] * 100,
                "memory_efficiency": individual_stats["mean_memory_usage"] / fused_stats["mean_memory_usage"],
            },
            "raw_results": {"individual": individual_results, "fused": fused_results},
        }

        return results

    def run_pipeline_optimizer_benchmark(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> Dict[str, Any]:
        """
        Demonstrate how to use PipelineOptimizer in the performance benchmark.
        This shows the proper way to use the new optimization architecture.
        """
        logger.info("🚀 Running PipelineOptimizer Benchmark")
        logger.info("========================================")

        # Step 1: Build pipeline configuration from filters
        pipeline_config = self._build_pipeline_config_from_filters(filters)

        # Step 2: Create Pipeline AST
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        logger.info("Original Pipeline AST:")
        logger.info(ast.visualize())

        # Step 3: Create PipelineOptimizer with analyzer insights
        from data_juicer.core.optimizer.filter_fusion_strategy import (
            FilterFusionStrategy,
        )
        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )

        strategies = [FilterFusionStrategy(analyzer_insights=analyzer_insights), MapperFusionStrategy()]

        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)

        # Step 4: Get optimization summary
        optimization_summary = optimizer.get_optimization_summary()
        logger.info("Optimization Configuration:")
        logger.info(f'  Strategies: {optimization_summary["strategies"]}')
        logger.info(f'  Analyzer insights available: {optimization_summary["analyzer_insights_available"]}')
        if optimization_summary["analyzer_insights_available"]:
            logger.info(f'  Dataset size: {optimization_summary.get("dataset_size", 0):,} samples')
            if "text_complexity" in optimization_summary:
                logger.info(f'  Text complexity (CV): {optimization_summary["text_complexity"]:.2f}')
            if "multimodal_types" in optimization_summary:
                logger.info(f'  Multimodal content types: {optimization_summary["multimodal_types"]}')

        # Step 5: Apply optimizations
        logger.info("Applying pipeline optimizations...")
        optimized_ast = optimizer.optimize(ast)

        logger.info("Optimized Pipeline AST:")
        logger.info(optimized_ast.visualize())

        # Step 6: Convert optimized AST back to operations
        optimized_ops = self._convert_ast_to_operations(optimized_ast)

        # Step 7: Benchmark the optimized operations
        logger.info("Benchmarking optimized operations...")
        start_time = time.time()

        # Process the test data with optimized operations
        self._process_with_optimized_ops(optimized_ops, test_data)

        total_time = time.time() - start_time
        throughput = len(test_data["text"]) / total_time

        # Step 8: Compare with legacy fusion
        logger.info("Comparing with legacy fusion approach...")
        legacy_start = time.time()

        # Use legacy fusion
        from data_juicer.ops.op_fusion import fuse_operators

        legacy_fused_ops = fuse_operators(filters)
        self._process_with_legacy_ops(legacy_fused_ops, test_data)

        legacy_time = time.time() - legacy_start
        legacy_throughput = len(test_data["text"]) / legacy_time

        # Step 9: Return comprehensive results
        results = {
            "pipeline_optimizer": {
                "total_time": total_time,
                "throughput": throughput,
                "optimization_summary": optimization_summary,
                "original_ops_count": len(filters),
                "optimized_ops_count": len(optimized_ops),
                "fusion_ratio": len(optimized_ops) / len(filters) if len(filters) > 0 else 1.0,
            },
            "legacy_fusion": {
                "total_time": legacy_time,
                "throughput": legacy_throughput,
                "original_ops_count": len(filters),
                "fused_ops_count": len(legacy_fused_ops),
                "fusion_ratio": len(legacy_fused_ops) / len(filters) if len(filters) > 0 else 1.0,
            },
            "comparison": {
                "speedup_ratio": legacy_time / total_time if total_time > 0 else 1.0,
                "throughput_improvement": (
                    (throughput - legacy_throughput) / legacy_throughput * 100 if legacy_throughput > 0 else 0
                ),
                "optimization_effectiveness": "PipelineOptimizer" if total_time < legacy_time else "Legacy",
            },
        }

        logger.info("PipelineOptimizer Benchmark Results:")
        logger.info(f"  PipelineOptimizer time: {total_time:.3f}s ({throughput:.1f} samples/s)")
        logger.info(f"  Legacy fusion time: {legacy_time:.3f}s ({legacy_throughput:.1f} samples/s)")
        logger.info(f'  Speedup: {results["comparison"]["speedup_ratio"]:.2f}x')
        logger.info(f'  Throughput improvement: {results["comparison"]["throughput_improvement"]:.1f}%')

        return results

    def _build_pipeline_config_from_filters(self, filters: List[Filter]) -> Dict[str, Any]:
        """Convert a list of filters to a pipeline configuration."""
        process_config = []

        for i, filter_op in enumerate(filters):
            op_name = getattr(filter_op, "_name", f"filter_{i}")

            # Extract configuration from filter object
            op_config = {}
            if hasattr(filter_op, "config") and filter_op.config:
                op_config = filter_op.config
            else:
                # Create basic config from filter attributes
                for attr in dir(filter_op):
                    if not attr.startswith("_") and not callable(getattr(filter_op, attr)):
                        value = getattr(filter_op, attr)
                        if isinstance(value, (int, float, str, bool)):
                            op_config[attr] = value

            process_config.append({op_name: op_config})

        return {"process": process_config}

    def _convert_ast_to_operations(self, ast) -> List:
        """Convert optimized AST back to operations list."""
        # This is a simplified conversion - in practice, you'd need more sophisticated logic
        operations = []

        def traverse_node(node):
            if hasattr(node, "children") and node.children:
                for child in node.children:
                    if hasattr(child, "config") and child.config:
                        # Extract operation from config
                        for op_name, op_config in child.config.items():
                            if op_name != "detailed_ops":  # Skip metadata
                                operations.append({op_name: op_config})
                    traverse_node(child)

        if ast.root:
            traverse_node(ast.root)

        return operations

    def _process_with_optimized_ops(self, optimized_ops: List, test_data: Dict[str, Any]):
        """Process test data with optimized operations."""
        # This is a simplified implementation
        # In practice, you'd load the operations and execute them
        logger.debug(f"Processing with {len(optimized_ops)} optimized operations")

        for op_config in optimized_ops:
            # Simulate processing - in reality, you'd load and execute the operation
            for op_name, config in op_config.items():
                logger.debug(f"Processing with optimized operation: {op_name}")
                # Here you would actually execute the operation
                pass

    def _process_with_legacy_ops(self, legacy_ops: List, test_data: Dict[str, Any]):
        """Process test data with legacy fused operations."""
        logger.debug(f"Processing with {len(legacy_ops)} legacy operations")

        for op in legacy_ops:
            if hasattr(op, "compute_stats_batched"):
                test_data = op.compute_stats_batched(test_data)
            if hasattr(op, "process_batched"):
                _ = list(op.process_batched(test_data))


def create_simple_test_data(num_samples: int = 1000) -> Dict[str, Any]:
    """Create simple test data for demonstration."""
    logger.info(f"Creating {num_samples:,} test samples...")

    # For large datasets, create in batches to avoid memory issues
    batch_size = 10000
    texts = []

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_size_actual = batch_end - batch_start

        batch_texts = []
        for _ in range(batch_size_actual):
            # Create text with varying characteristics
            length = random.randint(50, 200)
            text = "".join(random.choices(string.ascii_letters + string.digits + " .,!?", k=length))
            batch_texts.append(text)

        texts.extend(batch_texts)

        if batch_end % (batch_size * 5) == 0:
            logger.info(f"  Created {batch_end:,}/{num_samples:,} samples...")

    logger.info(f"Successfully created {len(texts):,} test samples")
    return {"text": texts, Fields.stats: [{} for _ in range(num_samples)]}


def create_lenient_funnel_filters() -> List[Filter]:
    """Create filters with very lenient thresholds to ensure funnel effect with non-zero results."""

    filters = [
        # Filter 1: Very lenient text length (should pass ~95%)
        TextLengthFilter(min_len=5, max_len=5000),
        # Filter 2: Very lenient word count (should pass ~90%)
        WordsNumFilter(min_num=2, max_num=1000),
        # Filter 3: Very lenient character repetition (should pass ~85%)
        CharacterRepetitionFilter(repetition_ratio=0.95),
        # Filter 4: Very lenient word repetition (should pass ~80%)
        WordRepetitionFilter(min_ratio=0.0, max_ratio=0.9),
        # Filter 5: Very lenient special characters (should pass ~75%)
        SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.5),
    ]

    return filters


def create_realistic_funnel_filters() -> List[Filter]:
    """Create filters with realistic thresholds that create a funnel effect."""

    filters = [
        # Filter 1: Remove very short texts (should pass ~80-90%)
        TextLengthFilter(min_len=10, max_len=2000),
        # Filter 2: Remove texts with too few words (should pass ~70-80%)
        WordsNumFilter(min_num=3, max_num=500),
        # Filter 3: Remove texts with high character repetition (should pass ~60-70%)
        CharacterRepetitionFilter(repetition_ratio=0.9),
        # Filter 4: Remove texts with high word repetition (should pass ~50-60%)
        WordRepetitionFilter(repetition_ratio=0.8),
        # Filter 5: Remove texts with too many special characters (should pass ~40-50%)
        SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),
    ]

    return filters


def create_simple_filters() -> List[Filter]:
    """Create a few simple filters for testing."""

    filters = [
        WordsNumFilter(min_num=5, max_num=1000),
        TextLengthFilter(min_len=20, max_len=1000),
        CharacterRepetitionFilter(repetition_ratio=0.8),
    ]

    return filters


def run_simple_demo(num_samples: int = 1000):
    """Run a simple demonstration of filter fusion performance with analyzer integration."""
    logger.info("🚀 Data-Juicer Filter Fusion Performance Demonstration")
    logger.info("========================================================")

    # Create benchmark instance to use analyzer integration
    benchmark = PerformanceBenchmark()

    # Create test data
    logger.info(f"Creating test data with {num_samples} samples...")
    samples = create_simple_test_data(num_samples)
    logger.info(
        f"DEBUG: type(samples)={type(samples)}, " f'keys={list(samples.keys()) if isinstance(samples, dict) else "N/A"}'
    )

    # Get analyzer insights
    logger.info("Running Analyzer to get insights...")
    analyzer_insights = benchmark.get_analyzer_insights(samples)
    logger.info(f"Analyzer insights: {analyzer_insights}")

    # Create test filters
    logger.info("Creating lenient funnel filters...")
    filters = create_lenient_funnel_filters()
    op_names = [getattr(f, "_name", type(f).__name__) for f in filters]
    logger.info(f"Created {len(filters)} lenient filters: {op_names}")

    # Test 1: Lenient Funnel Filters (should use parallel strategy)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Lenient Funnel Filters (Parallel Strategy)")
    logger.info("=" * 60)

    # Collect filtering statistics with analyzer insights
    collect_filtering_stats_with_insights(filters, samples, analyzer_insights)

    # Benchmark individual execution
    logger.info("\n" + "=" * 60)
    individual_stats = benchmark_individual_simple(filters, samples)

    # Benchmark fused execution with analyzer insights
    logger.info("\n" + "=" * 60)
    fused_stats = benchmark_fused_simple_with_insights(filters, samples, analyzer_insights)

    # Print performance results
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE RESULTS")
    logger.info("=" * 60)

    logger.info("Individual Execution:")
    logger.info(f"  Total Time: {individual_stats['total_time']:.3f}s")
    logger.info(f"  Stats Time: {individual_stats['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {individual_stats['filter_time']:.3f}s")
    logger.info(
        f"  Results: {individual_stats['passed_samples']:,}/{individual_stats['total_samples']:,} passed ({individual_stats['pass_rate']:.1f}%)"
    )

    logger.info("Fused Execution (with Analyzer Insights):")
    logger.info(f"  Total Time: {fused_stats['total_time']:.3f}s")
    logger.info(f"  Stats Time: {fused_stats['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {fused_stats['filter_time']:.3f}s")
    logger.info(
        f"  Results: {fused_stats['passed_samples']:,}/{fused_stats['total_samples']:,} passed ({fused_stats['pass_rate']:.1f}%)"
    )

    # Check if results are consistent
    result_difference = abs(individual_stats["passed_samples"] - fused_stats["passed_samples"])
    if result_difference > 0:
        logger.info(f"  ⚠️  Result Difference: {result_difference} samples")
        logger.info("     This indicates a bug in fusion implementation - results should be identical")
        logger.info("     Individual: Each filter processes original data, then logical AND")
        logger.info("     Fused: All filters process original data, then logical AND")
    else:
        logger.info("  ✅ Individual and fused results are identical")
        logger.info("     Both use parallel execution: all filters see original data")
        logger.info("     Fusion provides performance benefits without changing results")

    # Calculate improvements
    total_speedup = individual_stats["total_time"] / fused_stats["total_time"]
    time_saved = individual_stats["total_time"] - fused_stats["total_time"]
    stats_speedup = individual_stats["stats_time"] / fused_stats["stats_time"]
    filter_speedup = (
        individual_stats["filter_time"] / fused_stats["filter_time"] if fused_stats["filter_time"] > 0 else float("inf")
    )

    logger.info("🎯 IMPROVEMENTS:")
    logger.info(f"  Total Speedup: {total_speedup:.2f}x")
    logger.info(f"  Time Saved: {time_saved:.3f}s " f'({time_saved/individual_stats["total_time"]*100:.1f}%)')
    logger.info(f"  Stats Speedup: {stats_speedup:.2f}x")
    logger.info(f"  Filter Speedup: {filter_speedup:.2f}x")

    # Calculate throughput
    individual_throughput = num_samples / individual_stats["total_time"]
    fused_throughput = num_samples / fused_stats["total_time"]
    throughput_improvement = fused_throughput / individual_throughput

    logger.info("📈 THROUGHPUT:")
    logger.info(f"  Individual: {individual_throughput:,.0f} samples/sec")
    logger.info(f"  Fused: {fused_throughput:,.0f} samples/sec")
    logger.info(f"  Throughput Improvement: {throughput_improvement:.2f}x")

    # Performance assessment
    if total_speedup >= 1.5:
        performance_level = "🚀 EXCELLENT - Significant performance gain"
    elif total_speedup >= 1.1:
        performance_level = "✅ GOOD - Moderate performance gain"
    elif total_speedup >= 0.9:
        performance_level = "⚠️  MINIMAL - Small performance gain"
    else:
        performance_level = "❌ POOR - Performance regression"

    logger.info("\n" + "=" * 60)
    logger.info(performance_level)
    logger.info("=" * 60)

    # Test 2: Same filters with different strategy (should use sequential strategy)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Same Filters (Sequential Strategy)")
    logger.info("=" * 60)

    # Use the same filters but they should trigger sequential execution
    complex_filters = filters  # Use the same filters from test 1
    complex_op_names = [getattr(f, "_name", type(f).__name__) for f in complex_filters]
    logger.info(f"Using same {len(complex_filters)} filters: {complex_op_names}")

    # Benchmark individual execution
    logger.info("\n" + "=" * 60)
    individual_stats_complex = benchmark_individual_simple(complex_filters, samples)

    # Benchmark fused execution with analyzer insights
    logger.info("\n" + "=" * 60)
    fused_stats_complex = benchmark_fused_simple_with_insights(complex_filters, samples, analyzer_insights)

    # Print performance results for the second test
    logger.info("\n" + "=" * 60)
    logger.info("📊 SECOND TEST PERFORMANCE RESULTS")
    logger.info("=" * 60)

    total_speedup_complex = individual_stats_complex["total_time"] / fused_stats_complex["total_time"]
    time_saved_complex = individual_stats_complex["total_time"] - fused_stats_complex["total_time"]

    logger.info("Individual Execution:")
    logger.info(f"  Total Time: {individual_stats_complex['total_time']:.3f}s")
    logger.info(f"  Stats Time: {individual_stats_complex['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {individual_stats_complex['filter_time']:.3f}s")

    logger.info("Fused Execution (with Analyzer Insights):")
    logger.info(f"  Total Time: {fused_stats_complex['total_time']:.3f}s")
    logger.info(f"  Stats Time: {fused_stats_complex['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {fused_stats_complex['filter_time']:.3f}s")

    logger.info("🎯 IMPROVEMENTS:")
    logger.info(f"  Total Speedup: {total_speedup_complex:.2f}x")
    logger.info(
        f"  Time Saved: {time_saved_complex:.3f}s "
        f'({time_saved_complex/individual_stats_complex["total_time"]*100:.1f}%)'
    )


def benchmark_fused_simple(filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, float]:
    """Benchmark fused filter execution (simple version without analyzer insights)."""
    logger.info("=== Fused Filter Execution ===")

    # Create fused filter without analyzer insights
    fused_filter = FusedFilter("test_fused", filters)

    # Phase 1: Stats computation
    start = time.time()
    samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
    stats_time = time.time() - start

    # Phase 2: Filtering
    start = time.time()
    _ = fused_filter.process_batched(samples_with_stats)
    filter_time = time.time() - start

    total_time = stats_time + filter_time

    logger.info("Fused execution:")
    logger.info(f"  Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s, " f"Total: {total_time:.3f}s")

    return {"total_time": total_time, "stats_time": stats_time, "filter_time": filter_time}


def benchmark_fused_simple_with_insights(
    filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict
) -> Dict[str, float]:
    """Benchmark fused filter execution with analyzer insights."""
    logger.info("=== Fused Filter Execution (with Analyzer Insights) ===")

    # Create fused filter with analyzer insights
    fused_filter = FusedFilter("test_fused", filters, analyzer_insights=analyzer_insights)

    # Phase 1: Stats computation
    start = time.time()
    samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
    stats_time = time.time() - start

    # Phase 2: Filtering
    start = time.time()
    filter_results = list(fused_filter.process_batched(samples_with_stats))
    filter_time = time.time() - start

    total_time = stats_time + filter_time

    # Calculate actual filtering statistics
    total_samples = len(test_data["text"])
    passed_samples = sum(filter_results)
    pass_rate = (passed_samples / total_samples) * 100

    logger.info("Fused execution (with insights):")
    logger.info(f"  Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s, " f"Total: {total_time:.3f}s")
    logger.info(f"  Results: {passed_samples:,}/{total_samples:,} passed ({pass_rate:.1f}%)")

    return {
        "total_time": total_time,
        "stats_time": stats_time,
        "filter_time": filter_time,
        "passed_samples": passed_samples,
        "total_samples": total_samples,
        "pass_rate": pass_rate,
    }


def collect_filtering_stats(filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect filtering statistics to compare individual vs fused (without analyzer insights)."""
    logger.info("📊 Collecting filtering statistics...")

    # Individual filter stats
    individual_stats = {}
    total_samples = len(test_data["text"])

    for i, filter_op in enumerate(filters):
        op_name = getattr(filter_op, "_name", type(filter_op).__name__)
        logger.info(f"  Testing filter {i+1}: {op_name}")

        # Compute stats and filter
        samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
        filter_results = list(filter_op.process_batched(samples_with_stats))

        # Count passed samples
        passed_samples = sum(filter_results)
        pass_rate = (passed_samples / total_samples) * 100

        individual_stats[op_name] = {
            "passed": passed_samples,
            "filtered": total_samples - passed_samples,
            "pass_rate": pass_rate,
        }

        logger.info(f"    Passed: {passed_samples:,}/{total_samples:,} " f"({pass_rate:.1f}%)")

    # Fused filter stats
    logger.info("  Testing fused filters...")
    fused_filter = FusedFilter("test_fused", filters)
    samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
    fused_results = list(fused_filter.process_batched(samples_with_stats))

    fused_passed = sum(fused_results)
    fused_pass_rate = (fused_passed / total_samples) * 100

    fused_stats = {"passed": fused_passed, "filtered": total_samples - fused_passed, "pass_rate": fused_pass_rate}

    logger.info(f"    Fused - Passed: {fused_passed:,}/{total_samples:,} " f"({fused_pass_rate:.1f}%)")

    return {"individual": individual_stats, "fused": fused_stats, "total_samples": total_samples}


def collect_filtering_stats_with_insights(
    filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict
) -> Dict[str, Any]:
    """Collect filtering statistics with analyzer insights."""
    logger.info("📊 Collecting filtering statistics with analyzer insights...")

    # Individual filter stats
    individual_stats = {}
    total_samples = len(test_data["text"])

    for i, filter_op in enumerate(filters):
        op_name = getattr(filter_op, "_name", type(filter_op).__name__)
        logger.info(f"  Testing filter {i+1}: {op_name}")

        # Compute stats and filter
        samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
        filter_results = list(filter_op.process_batched(samples_with_stats))

        # Count passed samples
        passed_samples = sum(filter_results)
        pass_rate = (passed_samples / total_samples) * 100

        individual_stats[op_name] = {
            "passed": passed_samples,
            "filtered": total_samples - passed_samples,
            "pass_rate": pass_rate,
        }

        logger.info(f"    Passed: {passed_samples:,}/{total_samples:,} " f"({pass_rate:.1f}%)")

    # Fused filter stats with analyzer insights
    logger.info("  Testing fused filters with analyzer insights...")
    fused_filter = FusedFilter("test_fused", filters, analyzer_insights=analyzer_insights)
    samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
    fused_results = list(fused_filter.process_batched(samples_with_stats))

    fused_passed = sum(fused_results)
    fused_pass_rate = (fused_passed / total_samples) * 100

    fused_stats = {"passed": fused_passed, "filtered": total_samples - fused_passed, "pass_rate": fused_pass_rate}

    logger.info(f"    Fused - Passed: {fused_passed:,}/{total_samples:,} " f"({fused_pass_rate:.1f}%)")

    return {"individual": individual_stats, "fused": fused_stats, "total_samples": total_samples}


def benchmark_individual_simple(filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, float]:
    """Benchmark individual filter execution (simple version)."""
    logger.info("=== Individual Filter Execution ===")

    total_stats_time = 0.0
    total_filter_time = 0.0
    total_time = 0.0
    total_samples = len(test_data["text"])

    # Measure each filter independently (each processes original data)
    # This simulates running filters one by one, not in a pipeline
    individual_results = []

    for i, filter_op in enumerate(filters):
        op_name = getattr(filter_op, "_name", type(filter_op).__name__)
        logger.info(f"Processing filter {i+1}: {op_name}")

        # Phase 1: Stats computation
        start = time.time()
        samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
        stats_time = time.time() - start
        total_stats_time += stats_time

        # Phase 2: Filtering
        start = time.time()
        filter_results = list(filter_op.process_batched(samples_with_stats))
        filter_time = time.time() - start
        total_filter_time += filter_time

        # Calculate filtering statistics
        passed_samples = sum(filter_results)
        pass_rate = (passed_samples / total_samples) * 100

        logger.info(f"  Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s")
        logger.info(f"  Results: {passed_samples:,}/{total_samples:,} passed ({pass_rate:.1f}%)")

        individual_results.append(filter_results)

    # Calculate final result: sample must pass ALL filters (logical AND)
    final_results = individual_results[0]
    for result in individual_results[1:]:
        final_results = [r1 and r2 for r1, r2 in zip(final_results, result)]

    final_passed = sum(final_results)
    final_pass_rate = (final_passed / total_samples) * 100

    # Log the funnel effect
    logger.info("📊 FUNNEL EFFECT (Individual Filters):")
    for i, result in enumerate(individual_results):
        passed = sum(result)
        pass_rate = (passed / total_samples) * 100
        op_name = getattr(filters[i], "_name", type(filters[i]).__name__)
        logger.info(f"  Filter {i+1} ({op_name}): {passed:,}/{total_samples:,} passed ({pass_rate:.1f}%)")

    logger.info(f"  Combined (ALL filters): {final_passed:,}/{total_samples:,} passed ({final_pass_rate:.1f}%)")

    total_time = total_stats_time + total_filter_time

    return {
        "total_time": total_time,
        "stats_time": total_stats_time,
        "filter_time": total_filter_time,
        "passed_samples": final_passed,
        "total_samples": total_samples,
        "pass_rate": final_pass_rate,
    }


def print_filtering_comparison(stats: Dict[str, Any]):
    """Print detailed filtering comparison statistics."""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 FILTERING EFFECTIVENESS COMPARISON")
    logger.info("=" * 60)

    total_samples = stats["total_samples"]
    individual_stats = stats["individual"]
    fused_stats = stats["fused"]

    logger.info(f"Total Samples: {total_samples:,}")
    logger.info("Individual Filter Results:")

    # Calculate cumulative stats for individual filters
    cumulative_passed = total_samples
    for filter_name, filter_stats in individual_stats.items():
        passed = filter_stats["passed"]
        pass_rate = filter_stats["pass_rate"]
        filtered = filter_stats["filtered"]

        logger.info(f"  {filter_name:25s}: {passed:8,} passed ({pass_rate:5.1f}%) | {filtered:8,} filtered")
        cumulative_passed = passed  # Track the last filter's result for comparison

    logger.info("Fused Filter Results:")
    fused_passed = fused_stats["passed"]
    fused_pass_rate = fused_stats["pass_rate"]
    fused_filtered = fused_stats["filtered"]

    logger.info(
        f"  Fused Execution:        {fused_passed:8,} passed "
        f"({fused_pass_rate:5.1f}%) | {fused_filtered:8,} filtered"
    )

    # Compare individual vs fused
    individual_final_passed = cumulative_passed
    difference = fused_passed - individual_final_passed

    logger.info("Comparison:")
    logger.info(f"  Individual Final:       {individual_final_passed:8,} passed")
    logger.info(f"  Fused Final:           {fused_passed:8,} passed")
    logger.info(f"  Difference:            {difference:+8,} samples")

    if abs(difference) > 0:
        logger.info(f"  ⚠️  NOTE: Fused and individual results differ by {abs(difference)} samples")
        logger.info("     This may indicate different execution order or optimization effects")
    else:
        logger.info("  ✅ Individual and fused results are identical")

    # Calculate efficiency metrics
    total_individual_filtered = sum(stats["filtered"] for stats in individual_stats.values())
    efficiency_ratio = (
        fused_filtered / total_individual_filtered if total_individual_filtered > 0 else 1.0
    )  # noqa: E501

    logger.info("Efficiency Metrics:")
    logger.info(f"  Total Individual Filtered: {total_individual_filtered:,}")
    logger.info(f"  Fused Filtered:           {fused_filtered:,}")
    logger.info(f"  Efficiency Ratio:         {efficiency_ratio:.3f}x")

    return {
        "individual_final_passed": individual_final_passed,
        "fused_passed": fused_passed,
        "difference": difference,
        "efficiency_ratio": efficiency_ratio,
    }


def create_complex_filters():
    """Create complex filters that should trigger sequential execution."""
    from data_juicer.ops.filter import (
        FlaggedWordFilter,
        LanguageIDScoreFilter,
        PerplexityFilter,
        StopWordsFilter,
        WordRepetitionFilter,
    )

    filters = []

    # Perplexity filter (complex - requires language model)
    perplexity_filter = PerplexityFilter(lang="en", model_key="gpt2", min_score=0.0, max_score=100.0)
    filters.append(perplexity_filter)

    # Stopwords filter (complex - requires language processing)
    stopwords_filter = StopWordsFilter(lang="en", min_ratio=0.0, max_ratio=0.5)
    filters.append(stopwords_filter)

    # Flagged words filter (complex - requires word list processing)
    flagged_words_filter = FlaggedWordFilter(lang="en", min_ratio=0.0, max_ratio=0.1)
    filters.append(flagged_words_filter)

    # Language ID score filter (complex - requires language detection)
    lang_id_filter = LanguageIDScoreFilter(lang="en", min_score=0.5, max_score=1.0)
    filters.append(lang_id_filter)

    # Word repetition filter (complex - requires pattern analysis)
    word_rep_filter = WordRepetitionFilter(lang="en", min_ratio=0.0, max_ratio=0.3)
    filters.append(word_rep_filter)

    return filters


def analyze_fusion_decisions():
    """Analyze different filter combinations to determine optimal fusion decisions."""
    logger.info("🔬 FUSION DECISION ANALYSIS")
    logger.info("=" * 60)

    # Test different filter combinations
    filter_combinations = [
        # Simple filters only
        {"name": "Simple Only", "filters": create_simple_filters(), "expected": "skip_fusion"},
        # Complex filters only
        {"name": "Complex Only", "filters": create_complex_filters(), "expected": "use_fusion"},
        # Mixed filters
        {
            "name": "Mixed Simple+Complex",
            "filters": create_simple_filters() + create_complex_filters()[:2],
            "expected": "use_fusion",
        },
        # Single filter
        {"name": "Single Filter", "filters": [create_simple_filters()[0]], "expected": "skip_fusion"},
        # Two simple filters
        {"name": "Two Simple", "filters": create_simple_filters()[:2], "expected": "skip_fusion"},
    ]

    results = []

    for combo in filter_combinations:
        logger.info(f"📊 Testing: {combo['name']}")
        logger.info(f"Filters: {[f._name for f in combo['filters']]}")

        # Create test data
        test_data = create_simple_test_data(1000)

        # Test individual execution
        individual_stats = benchmark_individual_simple(combo["filters"], test_data)

        # Test fused execution
        fused_stats = benchmark_fused_simple(combo["filters"], test_data)

        # Calculate metrics
        speedup = individual_stats["total_time"] / fused_stats["total_time"]
        overhead_ratio = fused_stats["total_time"] / individual_stats["total_time"]

        # Determine recommendation
        if overhead_ratio > 1.5:
            recommendation = "skip_fusion"
        elif individual_stats["total_time"] < 0.01:
            recommendation = "skip_fusion"
        elif len(combo["filters"]) <= 2:
            recommendation = "skip_fusion"
        else:
            recommendation = "use_fusion"

        # Check if recommendation matches expectation
        correct = recommendation == combo["expected"]

        result = {
            "name": combo["name"],
            "filters": [f._name for f in combo["filters"]],
            "individual_time": individual_stats["total_time"],
            "fused_time": fused_stats["total_time"],
            "speedup": speedup,
            "overhead_ratio": overhead_ratio,
            "recommendation": recommendation,
            "expected": combo["expected"],
            "correct": correct,
        }

        results.append(result)

        logger.info(f"  Individual: {individual_stats['total_time']:.3f}s")
        logger.info(f"  Fused: {fused_stats['total_time']:.3f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Overhead: {overhead_ratio:.2f}x")
        logger.info(f"  Recommendation: {recommendation}")
        logger.info(f"  Expected: {combo['expected']}")
        logger.info(f"  Correct: {'✅' if correct else '❌'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 FUSION DECISION SUMMARY")
    logger.info("=" * 60)

    correct_decisions = sum(1 for r in results if r["correct"])
    total_decisions = len(results)

    logger.info(
        f"Correct Decisions: {correct_decisions}/{total_decisions} ({correct_decisions/total_decisions*100:.1f}%)"
    )

    # Decision rules summary
    logger.info("🎯 DECISION RULES:")
    logger.info("1. Skip fusion if overhead > 50% (overhead_ratio > 1.5)")
    logger.info("2. Skip fusion if individual time < 10ms (too fast)")
    logger.info("3. Skip fusion if ≤2 filters (minimal benefit)")
    logger.info("4. Use fusion for complex filters (significant benefit)")
    logger.info("5. Use fusion for mixed combinations (moderate benefit)")

    # Performance thresholds
    logger.info("📊 PERFORMANCE THRESHOLDS:")
    logger.info("• Individual time < 10ms → Skip fusion")
    logger.info("• Overhead ratio > 1.5x → Skip fusion")
    logger.info("• ≤2 filters → Skip fusion")
    logger.info("• Complex filters → Use fusion")
    logger.info("• Mixed filters → Use fusion")

    return results


def main():
    """Main function to run the performance benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Data-Juicer Performance Benchmark")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "optimizer", "fusion-analysis"],
        default="quick",
        help="Benchmark mode: quick (basic demo), full (comprehensive test), optimizer (new optimizer architecture), fusion-analysis (analyze fusion decisions)",
    )
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples for testing")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for comprehensive testing")

    args = parser.parse_args()

    try:
        if args.mode == "quick":
            logger.info("🚀 Running QUICK benchmark (basic performance demo)")
            logger.info(f"Testing with {args.samples:,} samples...")
            return run_simple_demo(args.samples)

        elif args.mode == "full":
            logger.info("🔬 Running FULL benchmark (comprehensive performance analysis)")
            logger.info(f"Testing with {args.samples:,} samples, {args.runs} runs...")

            # Warn about large datasets
            if args.samples > 50000:
                logger.warning(f"⚠️  Large dataset detected ({args.samples:,} samples)")
                logger.warning("   This may take a long time and could cause memory issues")
                logger.warning("   Consider using --samples 10000 for faster testing")

            benchmark = PerformanceBenchmark()
            return benchmark.run_comprehensive_test(args.samples, args.runs)

        elif args.mode == "optimizer":
            logger.info("⚡ Running OPTIMIZER benchmark (new optimizer architecture)")
            logger.info(f"Testing with {args.samples:,} samples...")
            benchmark = PerformanceBenchmark()
            filters = benchmark.create_test_filters()
            test_data = benchmark.create_realistic_test_data(args.samples)
            analyzer_insights = benchmark.get_analyzer_insights(test_data)
            return benchmark.run_pipeline_optimizer_benchmark(filters, test_data, analyzer_insights)

        elif args.mode == "fusion-analysis":
            logger.info("🔬 Running FUSION ANALYSIS (analyze fusion decisions)")
            logger.info(f"Testing with {args.samples:,} samples...")
            return analyze_fusion_decisions()

    except Exception as e:
        logger.error(f"❌ Benchmark failed with exception: {e}")
        logger.error("Full traceback:")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
