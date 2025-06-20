#!/usr/bin/env python3
"""
Performance test to demonstrate the benefits of fused filter execution.
This test compares individual vs fused filter performance with real data.
"""

import random
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from loguru import logger

from data_juicer.core.optimizer.fused_op import FusedFilter
from data_juicer.ops.base_op import Filter
from data_juicer.ops.filter import (
    AlphanumericFilter,
    AverageLineLengthFilter,
    CharacterRepetitionFilter,
    FlaggedWordFilter,
    MaximumLineLengthFilter,
    PerplexityFilter,
    SpecialCharactersFilter,
    StopWordsFilter,
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

    def create_test_filters(self) -> List[Filter]:
        """Create a comprehensive set of test filters."""
        logger.info("Creating test filters...")

        # Import filter classes
        from data_juicer.ops.filter import (
            AlphanumericFilter,
            AverageLineLengthFilter,
            CharacterRepetitionFilter,
            FlaggedWordFilter,
            MaximumLineLengthFilter,
            PerplexityFilter,
            SpecialCharactersFilter,
            StopWordsFilter,
            TextLengthFilter,
            WordRepetitionFilter,
            WordsNumFilter,
        )

        filters = [
            WordsNumFilter(min_num=5, max_num=1000),
            TextLengthFilter(min_len=20, max_len=1000),
            CharacterRepetitionFilter(repetition_ratio=0.8),
            WordRepetitionFilter(min_ratio=0.0, max_ratio=0.5),
            SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),
            PerplexityFilter(max_ppl=1500),
            StopWordsFilter(min_ratio=0.1),
            FlaggedWordFilter(max_ratio=0.05),
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

    def run_individual_filters_benchmark(self, filters: List[Filter], test_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark individual filter execution."""
        logger.info("Running individual filters benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        total_stats_time = 0.0
        total_filter_time = 0.0

        for i, filter_op in enumerate(filters):
            logger.info(f"  Processing filter {i+1}/{len(filters)}: {filter_op._name}")

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

    def run_fused_filters_benchmark(self, filters: List[Filter], test_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark fused filter execution."""
        logger.info("Running fused filters benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Create fused filter
        fused_filter = FusedFilter("performance_test_fused", filters)

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

    def collect_filtering_statistics(self, filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect filtering statistics for comprehensive analysis."""
        logger.info("📊 Collecting comprehensive filtering statistics...")

        # Individual filter stats
        individual_stats = {}
        total_samples = len(test_data["text"])

        for i, filter_op in enumerate(filters):
            logger.info(f"  Testing filter {i+1}/{len(filters)}: {filter_op._name}")

            # Compute stats and filter
            samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
            filter_results = list(filter_op.process_batched(samples_with_stats))

            # Count passed samples
            passed_samples = sum(filter_results)
            pass_rate = (passed_samples / total_samples) * 100

            individual_stats[filter_op._name] = {
                "passed": passed_samples,
                "filtered": total_samples - passed_samples,
                "pass_rate": pass_rate,
            }

            logger.info(f"    Passed: {passed_samples:,}/{total_samples:,} ({pass_rate:.1f}%)")

        # Fused filter stats
        logger.info("  Testing fused filters...")
        fused_filter = FusedFilter("comprehensive_test_fused", filters)
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

        # Create test filters
        logger.info("Creating test filters...")
        filters = self.create_test_filters()

        # Collect filtering statistics first
        filtering_stats = self.collect_filtering_statistics(filters, test_data)
        comparison_results = print_filtering_comparison(filtering_stats)

        # Run multiple iterations for statistical significance
        individual_results = []
        fused_results = []

        for run in range(num_runs):
            logger.info(f"\n--- Run {run + 1}/{num_runs} ---")

            # Individual execution
            individual_result = self.run_individual_filters_benchmark(filters, test_data.copy())
            individual_results.append(individual_result)

            # Fused execution
            fused_result = self.run_fused_filters_benchmark(filters, test_data.copy())
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

        logger.info("\nPerformance Comparison:")
        logger.info(
            f"  Individual Execution: {results['individual']['mean_total_time']:.3f}s "
            f"± {results['individual']['std_total_time']:.3f}s"
        )
        logger.info(
            f"  Fused Execution: {results['fused']['mean_total_time']:.3f}s "
            f"± {results['fused']['std_total_time']:.3f}s"
        )

        logger.info("\nImprovements:")
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

        logger.info(f"\nAssessment: {assessment}")
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
        stats_speedup = individual_stats["mean_stats_time"] / fused_stats["mean_stats_time"]
        filter_speedup = individual_stats["mean_filter_time"] / fused_stats["mean_filter_time"]
        throughput_improvement = fused_stats["mean_throughput"] / individual_stats["mean_throughput"]

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
                "throughput_improvement": throughput_improvement,
                "time_saved_percent": (1 - fused_stats["mean_total_time"] / individual_stats["mean_total_time"]) * 100,
                "memory_efficiency": individual_stats["mean_memory_usage"] / fused_stats["mean_memory_usage"],
            },
            "raw_results": {"individual": individual_results, "fused": fused_results},
        }

        return results


def create_simple_test_data(num_samples: int = 1000) -> Dict[str, Any]:
    """Create simple test data for demonstration."""
    texts = []
    for _ in range(num_samples):
        # Create text with varying characteristics
        length = random.randint(50, 200)
        text = "".join(random.choices(string.ascii_letters + string.digits + " .,!?", k=length))
        texts.append(text)

    return {"text": texts, Fields.stats: [{} for _ in range(num_samples)]}


def create_simple_filters() -> List[Filter]:
    """Create a few simple filters for testing."""

    filters = [
        WordsNumFilter(min_num=5, max_num=1000),
        TextLengthFilter(min_len=20, max_len=1000),
        CharacterRepetitionFilter(repetition_ratio=0.8),
    ]

    return filters


def run_simple_demo():
    """Run a simple demonstration of filter fusion performance."""
    logger.info("🚀 Data-Juicer Filter Fusion Performance Demonstration")
    logger.info("========================================================")

    # Create test data
    num_samples = 1000
    logger.info(f"Creating test data with {num_samples} samples...")
    samples = create_simple_test_data(num_samples)
    logger.info(
        f"DEBUG: type(samples)={type(samples)}, " f'keys={list(samples.keys()) if isinstance(samples, dict) else "N/A"}'
    )

    # Create test filters
    logger.info("Creating test filters...")
    filters = create_simple_filters()
    logger.info(f"Created {len(filters)} filters: {[f._name for f in filters]}")

    # Test 1: Simple filters (should use parallel strategy)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Simple Filters (Parallel Strategy)")
    logger.info("=" * 60)

    # Collect filtering statistics
    collect_filtering_stats(filters, samples)

    # Benchmark individual execution
    logger.info("\n" + "=" * 60)
    individual_stats = benchmark_individual_simple(filters, samples)

    # Benchmark fused execution
    logger.info("\n" + "=" * 60)
    fused_stats = benchmark_fused_simple(filters, samples)

    # Print performance results
    logger.info("\n" + "=" * 60)
    logger.info("📊 PERFORMANCE RESULTS")
    logger.info("=" * 60)

    logger.info("Individual Execution:")
    logger.info(f"  Total Time: {individual_stats['total_time']:.3f}s")
    logger.info(f"  Stats Time: {individual_stats['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {individual_stats['filter_time']:.3f}s")

    logger.info("\nFused Execution:")
    logger.info(f"  Total Time: {fused_stats['total_time']:.3f}s")
    logger.info(f"  Stats Time: {fused_stats['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {fused_stats['filter_time']:.3f}s")

    # Calculate improvements
    total_speedup = individual_stats["total_time"] / fused_stats["total_time"]
    time_saved = individual_stats["total_time"] - fused_stats["total_time"]
    stats_speedup = individual_stats["stats_time"] / fused_stats["stats_time"]
    filter_speedup = (
        individual_stats["filter_time"] / fused_stats["filter_time"] if fused_stats["filter_time"] > 0 else float("inf")
    )

    logger.info("\n🎯 IMPROVEMENTS:")
    logger.info(f"  Total Speedup: {total_speedup:.2f}x")
    logger.info(f"  Time Saved: {time_saved:.3f}s " f'({time_saved/individual_stats["total_time"]*100:.1f}%)')
    logger.info(f"  Stats Speedup: {stats_speedup:.2f}x")
    logger.info(f"  Filter Speedup: {filter_speedup:.2f}x")

    # Calculate throughput
    individual_throughput = num_samples / individual_stats["total_time"]
    fused_throughput = num_samples / fused_stats["total_time"]
    throughput_improvement = fused_throughput / individual_throughput

    logger.info("\n📈 THROUGHPUT:")
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

    # Test 2: Complex filters (should use sequential strategy)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Complex Filters (Sequential Strategy)")
    logger.info("=" * 60)

    # Create complex filters
    complex_filters = create_complex_filters()
    logger.info(f"Created {len(complex_filters)} complex filters: " f"{[f._name for f in complex_filters]}")

    # Benchmark individual execution
    logger.info("\n" + "=" * 60)
    individual_stats_complex = benchmark_individual_simple(complex_filters, samples)

    # Benchmark fused execution
    logger.info("\n" + "=" * 60)
    fused_stats_complex = benchmark_fused_simple(complex_filters, samples)

    # Print performance results for complex filters
    logger.info("\n" + "=" * 60)
    logger.info("📊 COMPLEX FILTERS PERFORMANCE RESULTS")
    logger.info("=" * 60)

    total_speedup_complex = individual_stats_complex["total_time"] / fused_stats_complex["total_time"]
    time_saved_complex = individual_stats_complex["total_time"] - fused_stats_complex["total_time"]

    logger.info("Individual Execution:")
    logger.info(f"  Total Time: {individual_stats_complex['total_time']:.3f}s")
    logger.info(f"  Stats Time: {individual_stats_complex['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {individual_stats_complex['filter_time']:.3f}s")

    logger.info("\nFused Execution:")
    logger.info(f"  Total Time: {fused_stats_complex['total_time']:.3f}s")
    logger.info(f"  Stats Time: {fused_stats_complex['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {fused_stats_complex['filter_time']:.3f}s")

    logger.info("\n🎯 IMPROVEMENTS:")
    logger.info(f"  Total Speedup: {total_speedup_complex:.2f}x")
    logger.info(
        f"  Time Saved: {time_saved_complex:.3f}s "
        f'({time_saved_complex/individual_stats_complex["total_time"]*100:.1f}%)'
    )


def benchmark_individual_simple(filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, float]:
    """Benchmark individual filter execution (simple version)."""
    logger.info("=== Individual Filter Execution ===")

    total_stats_time = 0.0
    total_filter_time = 0.0
    total_time = 0.0

    for i, filter_op in enumerate(filters):
        logger.info(f"Processing filter {i+1}: {filter_op._name}")

        # Phase 1: Stats computation
        start = time.time()
        samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
        stats_time = time.time() - start
        total_stats_time += stats_time

        # Phase 2: Filtering
        start = time.time()
        _ = filter_op.process_batched(samples_with_stats)
        filter_time = time.time() - start
        total_filter_time += filter_time

        logger.info(f"  Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s")

    total_time = total_stats_time + total_filter_time

    return {"total_time": total_time, "stats_time": total_stats_time, "filter_time": total_filter_time}


def benchmark_fused_simple(filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, float]:
    """Benchmark fused filter execution (simple version)."""
    logger.info("=== Fused Filter Execution ===")

    # Create fused filter
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


def collect_filtering_stats(filters: List[Filter], test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Collect filtering statistics to compare individual vs fused."""
    logger.info("📊 Collecting filtering statistics...")

    # Individual filter stats
    individual_stats = {}
    total_samples = len(test_data["text"])

    for i, filter_op in enumerate(filters):
        logger.info(f"  Testing filter {i+1}: {filter_op._name}")

        # Compute stats and filter
        samples_with_stats = filter_op.compute_stats_batched(test_data.copy())
        filter_results = list(filter_op.process_batched(samples_with_stats))

        # Count passed samples
        passed_samples = sum(filter_results)
        pass_rate = (passed_samples / total_samples) * 100

        individual_stats[filter_op._name] = {
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
        cumulative_passed = passed  # Each filter processes the output of the previous

    logger.info("\nFused Filter Results:")
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

    logger.info("\nComparison:")
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

    logger.info("\nEfficiency Metrics:")
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


def create_test_filters():
    """Create a comprehensive set of test filters."""
    logger.info("Creating test filters...")

    filters = [
        WordsNumFilter(min_num=5, max_num=1000),
        TextLengthFilter(min_len=20, max_len=1000),
        CharacterRepetitionFilter(repetition_ratio=0.8),
        WordRepetitionFilter(min_ratio=0.0, max_ratio=0.5),
        SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),
        PerplexityFilter(max_ppl=1500),
        StopWordsFilter(min_ratio=0.1),
        FlaggedWordFilter(max_ratio=0.05),
        AlphanumericFilter(min_ratio=0.3),
        AverageLineLengthFilter(min_len=10, max_len=100),
        MaximumLineLengthFilter(min_len=10, max_len=200),
    ]

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
        logger.info(f"\n📊 Testing: {combo['name']}")
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
    logger.info("\n🎯 DECISION RULES:")
    logger.info("1. Skip fusion if overhead > 50% (overhead_ratio > 1.5)")
    logger.info("2. Skip fusion if individual time < 10ms (too fast)")
    logger.info("3. Skip fusion if ≤2 filters (minimal benefit)")
    logger.info("4. Use fusion for complex filters (significant benefit)")
    logger.info("5. Use fusion for mixed combinations (moderate benefit)")

    # Performance thresholds
    logger.info("\n📊 PERFORMANCE THRESHOLDS:")
    logger.info("• Individual time < 10ms → Skip fusion")
    logger.info("• Overhead ratio > 1.5x → Skip fusion")
    logger.info("• ≤2 filters → Skip fusion")
    logger.info("• Complex filters → Use fusion")
    logger.info("• Mixed filters → Use fusion")

    return results


def main():
    """Main function to run the performance benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Filter fusion performance benchmark")
    parser.add_argument("--mode", choices=["demo", "comprehensive", "analysis"], default="demo", help="Benchmark mode")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples for testing")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for comprehensive testing")

    args = parser.parse_args()

    if args.mode == "demo":
        logger.info("Running in DEMO mode (quick performance demonstration)")
        return run_simple_demo()
    elif args.mode == "analysis":
        logger.info("Running in ANALYSIS mode (fusion decision analysis)")
        return analyze_fusion_decisions()
    else:
        logger.info(f"Running in COMPREHENSIVE mode with {args.samples:,} samples, " f"{args.runs} runs")
        benchmark = PerformanceBenchmark()
        return benchmark.run_comprehensive_test(args.samples, args.runs)


if __name__ == "__main__":
    main()
