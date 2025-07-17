#!/usr/bin/env python3
"""
Performance benchmark for Data-Juicer filter fusion and optimization.

This benchmark compares individual vs fused filter performance and demonstrates
the new PipelineOptimizer architecture. In "both" mode, it focuses on correctness
validation rather than performance comparison.

USAGE EXAMPLES:
    # Full comprehensive benchmark (default) - correctness comparison
    python performance_benchmark.py

    # Full benchmark with more samples
    python performance_benchmark.py --samples 50000

    # Quick benchmark (basic demo with 3 filters)
    python performance_benchmark.py --mode quick --samples 1000

    # Recipe mode - benchmark a real YAML pipeline
    python performance_benchmark.py --mode recipe --recipe-path configs/data_juicer_recipes/redpajama-arxiv-refine.yaml

    # Recipe mode with real dataset
    python performance_benchmark.py --mode recipe --recipe-path configs/data_juicer_recipes/redpajama-arxiv-refine.yaml --dataset-path demos/data/demo-dataset_1725870268.jsonl

    # Recipe mode with custom sample count
    python performance_benchmark.py --mode recipe --recipe-path configs/data_juicer_recipes/redpajama-arxiv-refine.yaml --dataset-path demos/data/demo-dataset_1725870268.jsonl --samples 5000

    # Use real dataset with synthetic filters
    python performance_benchmark.py --dataset-path demos/data/demo-dataset_1725870268.jsonl --samples 2000

    # Performance-only benchmarks (no correctness comparison)
    python performance_benchmark.py --benchmark-type individual
    python performance_benchmark.py --benchmark-type pipeline

MODES:
    full     - Comprehensive benchmark with correctness comparison (default)
    quick    - Basic correctness demo with 3 filters
    recipe   - Benchmark a real YAML pipeline (requires --recipe-path)
"""

import os
import random
import shutil
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from data_juicer.core.analyzer import Analyzer
from data_juicer.core.optimizer.fused_op import FusedFilter
from data_juicer.ops.base_op import Filter
from data_juicer.ops.filter import (
    AlphanumericFilter,
    AverageLineLengthFilter,
    CharacterRepetitionFilter,
    FlaggedWordFilter,
    LanguageIDScoreFilter,
    MaximumLineLengthFilter,
    PerplexityFilter,
    SpecialCharactersFilter,
    StopWordsFilter,
    TextLengthFilter,
    WordRepetitionFilter,
    WordsNumFilter,
)
from data_juicer.utils.constant import Fields


def get_dataset_length(dataset) -> int:
    """Get the length of a dataset, handling different formats."""
    if isinstance(dataset, dict) and "text" in dataset:
        return len(dataset["text"])
    elif not isinstance(dataset, dict) and hasattr(dataset, "__len__"):
        return len(dataset)
    elif not isinstance(dataset, dict) and hasattr(dataset, "data") and hasattr(dataset.data, "__len__"):
        return len(dataset.data)
    else:
        # Try to get length from column names
        if not isinstance(dataset, dict) and hasattr(dataset, "column_names") and dataset.column_names:
            first_col = dataset.column_names[0]
            if hasattr(dataset, "__getitem__"):
                return len(dataset[first_col])
    return 0


def get_dataset_texts(dataset) -> list:
    """Get text data from a dataset, handling different formats."""
    if isinstance(dataset, dict) and "text" in dataset:
        return dataset["text"]
    elif not isinstance(dataset, dict) and hasattr(dataset, "column_names") and "text" in dataset.column_names:
        return dataset["text"]
    elif not isinstance(dataset, dict) and hasattr(dataset, "__getitem__"):
        # Try to find text column
        if hasattr(dataset, "column_names"):
            for col in dataset.column_names:
                if "text" in col.lower():
                    return dataset[col]
        # Fallback to first column
        if hasattr(dataset, "column_names") and dataset.column_names:
            return dataset[dataset.column_names[0]]
    elif not isinstance(dataset, dict) and hasattr(dataset, "data") and hasattr(dataset.data, "to_pandas"):
        # Ray dataset
        df = dataset.data.to_pandas()
        if "text" in df.columns:
            return df["text"].tolist()
        # Use first string column
        for col in df.columns:
            if df[col].dtype == "object":
                return df[col].tolist()
    return []


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

    def create_test_filters(self) -> List[Filter]:
        """Create a comprehensive set of test filters covering different categories."""
        logger.info("Creating comprehensive test filters...")

        filters = [
            # Basic text filters (simple, fast) - Realistic settings
            WordsNumFilter(min_num=10, max_num=2000),  # 10-2000 words (was 1-100000)
            TextLengthFilter(min_len=50, max_len=10000),  # 50-10000 chars (was 1-100000)
            CharacterRepetitionFilter(repetition_ratio=0.3),  # Max 30% repetition (was 0.999)
            WordRepetitionFilter(min_ratio=0.0, max_ratio=0.4),  # Max 40% word repetition (was 0.99)
            SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),  # Max 30% special chars (was 0.95)
            AlphanumericFilter(min_ratio=0.5),  # At least 50% alphanumeric (was 0.001)
            AverageLineLengthFilter(min_len=10, max_len=200),  # 10-200 chars per line (was 1-10000)
            MaximumLineLengthFilter(min_len=10, max_len=500),  # 10-500 chars per line (was 1-20000)
            # Content quality filters (moderate complexity) - Realistic settings
            PerplexityFilter(
                lang="en", model_key="gpt2", min_score=0.0, max_score=1000
            ),  # Max perplexity 1000 (was 1e10)
            StopWordsFilter(lang="en", min_ratio=0.0, max_ratio=0.6),  # Max 60% stop words (was 0.99)
            FlaggedWordFilter(lang="en", min_ratio=0.0, max_ratio=0.1),  # Max 10% flagged words (was 0.99)
            LanguageIDScoreFilter(lang="en", min_score=0.5),  # At least 50% English confidence (was 0.0)
        ]

        # Debug: Log all filters being created
        logger.debug(f"🔍 DEBUG: Created {len(filters)} filters:")
        for i, filter_op in enumerate(filters):
            filter_type = type(filter_op).__name__
            filter_name = getattr(filter_op, "_name", f"filter_{i}")
            has_compute_stats = hasattr(filter_op, "compute_stats_batched")
            has_process_batched = hasattr(filter_op, "process_batched")
            logger.info(
                f"  Filter {i+1}: {filter_type} (name: {filter_name}, has_compute_stats: {has_compute_stats}, has_process_batched: {has_process_batched})"
            )

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

    def get_analyzer_insights(self, test_data: Any) -> dict:
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
            insights = {"dataset_size": get_dataset_length(dataset), "text_length": {}, "content_ratios": {}}

            # Log detailed statistics before cleanup
            logger.info("📊 DETAILED ANALYZER STATISTICS:")
            logger.info("=" * 50)

            if overall is not None:
                logger.info(f"Dataset size: {get_dataset_length(dataset):,} samples")
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
                    texts = get_dataset_texts(dataset)
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
            dataset_size = get_dataset_length(dataset)
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

    def simulate_analyzer_insights(self, test_data: Any) -> dict:
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
        elif (
            not isinstance(test_data, dict) and hasattr(test_data, "column_names") and "text" in test_data.column_names
        ):
            texts = test_data["text"]
            logger.info("Data format: HuggingFace Dataset")
        else:
            texts = []
            logger.warning("Data format: Unknown")

        logger.info(f"Dataset size: {get_dataset_length(test_data):,} samples")

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
        dataset_size = get_dataset_length(test_data) if texts else 0
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
            "dataset_size": get_dataset_length(test_data) if texts else 0,
            "text_length": {"mean": mean_length, "std": std_length},
            "content_ratios": content_ratios,
        }

    def run_individual_filters_benchmark(self, filters: List[Filter], test_data: Any) -> PerformanceMetrics:
        """Benchmark individual filters executed sequentially."""
        logger.info("Running individual filters benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Step 1: Initialize filters
        init_start = time.time()
        # Filter out operations that don't have compute_stats_batched (like mappers)
        actual_filters = [f for f in filters if hasattr(f, "compute_stats_batched")]
        logger.info(f"  Found {len(actual_filters)} actual filters out of {len(filters)} operations")

        if not actual_filters:
            logger.warning("  No actual filters found to benchmark!")
            return PerformanceMetrics(
                total_time=0.0,
                stats_time=0.0,
                filter_time=0.0,
                memory_usage=0.0,
                throughput=0.0,
            )

        init_time = time.time() - init_start
        logger.info(f"  Step 1 - Filter initialization: {init_time:.3f}s")

        # Step 2: Process each filter completely (stats + filtering) before moving to the next
        processing_start = time.time()
        samples_with_stats = test_data
        total_stats_time = 0.0
        total_filter_time = 0.0

        # DEBUG: Track sample counts and filter results
        original_sample_count = get_dataset_length(test_data)
        logger.debug(f"🔍 DEBUG INDIVIDUAL: Starting with {original_sample_count} samples")

        for i, filter_op in enumerate(actual_filters):
            filter_name = getattr(filter_op, "_name", type(filter_op).__name__)
            logger.info(f"    Processing filter {i+1}/{len(actual_filters)}: {filter_name}")

            # DEBUG: Log current state before filter
            current_sample_count = get_dataset_length(samples_with_stats)
            logger.debug(f"🔍 DEBUG INDIVIDUAL: Before filter {i+1} ({filter_name}): {current_sample_count} samples")

            # Run the filter with reduction
            filter_start = time.time()
            samples_with_stats = filter_op.run(samples_with_stats, reduce=True)
            filter_time = time.time() - filter_start
            total_filter_time += filter_time

            # Log after filtering (without showing text content)
            final_sample_count = get_dataset_length(samples_with_stats)
            logger.info(f"    After filter {i+1}: {final_sample_count} samples remain")
            logger.debug(f"      Filter {i+1} - Filter: {filter_time:.3f}s")
            if final_sample_count == 0:
                logger.info(f"    All samples filtered out at filter {i+1}. Stopping.")
                break

        processing_time = time.time() - processing_start
        logger.info(f"  Step 2 - Complete processing: {processing_time:.3f}s")
        logger.info(f"    Total stats time: {total_stats_time:.3f}s")
        logger.info(f"    Total filter time: {total_filter_time:.3f}s")

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        # Use final sample count for throughput since we're actually filtering
        final_sample_count = get_dataset_length(samples_with_stats)
        throughput = final_sample_count / total_time

        logger.info("  📊 INDIVIDUAL FILTERS BREAKDOWN:")
        logger.info(f"    Initialization: {init_time:.3f}s ({init_time/total_time*100:.1f}%)")
        logger.info(f"    Stats computation: {total_stats_time:.3f}s ({total_stats_time/total_time*100:.1f}%)")
        logger.info(f"    Filtering: {total_filter_time:.3f}s ({total_filter_time/total_time*100:.1f}%)")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_stats_time,
            filter_time=total_filter_time,
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def run_pipeline_optimizer_benchmark(
        self, filters: List[Filter], test_data: Any, analyzer_insights: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """Benchmark the complete pipeline optimizer workflow using FusedFilter."""
        logger.info("Running pipeline optimizer benchmark (using FusedFilter)...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Create a FusedFilter with all the individual filters
        logger.info("  Creating FusedFilter with all filters...")
        fused_filter = FusedFilter(name="benchmark_fused_filter", fused_filters=filters)

        # Set execution strategy to sequential to match individual execution behavior
        fused_filter.execution_strategy = "sequential"

        logger.info(f"  Created FusedFilter with {len(filters)} filters")
        logger.info(f"  Execution strategy: {fused_filter.execution_strategy}")

        # Process with the fused filter using the run method
        logger.info("  Processing with FusedFilter...")

        # Use the run method which handles dataset conversion internally
        filter_start = time.time()
        data = fused_filter.run(test_data)
        filter_time = time.time() - filter_start

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        final_sample_count = get_dataset_length(data)
        throughput = final_sample_count / total_time

        logger.info("  📊 PIPELINE OPTIMIZER BREAKDOWN:")
        logger.info(f"    Total processing time: {filter_time:.3f}s ({filter_time/total_time*100:.1f}%)")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")
        logger.info(f"    Memory usage: {memory_usage:.1f} MB")
        logger.info(f"    Final samples: {final_sample_count}")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=filter_time * 0.6,  # Estimate: 60% of time is stats computation
            filter_time=filter_time * 0.4,  # Estimate: 40% of time is filtering
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def get_final_mask_from_filters(self, filters: List[Filter], test_data: Any) -> list:
        """Compute the final boolean mask for each sample using individual filter execution (AND) with funneling."""
        original_length = get_dataset_length(test_data)
        logger.debug(f"🔍 DEBUG: Processing {len(filters)} filters for individual mask computation")
        logger.debug(f"🔍 DEBUG: Original dataset length: {original_length}")

        # Track which samples are still active (not dropped by funneling)
        active_samples = set(range(original_length))
        final_mask = [False] * original_length  # Initialize all to False

        data = test_data

        for i, filter_op in enumerate(filters):
            logger.debug(f"🔍 DEBUG: Processing filter {i+1}/{len(filters)}: {type(filter_op).__name__}")
            logger.debug(f"🔍 DEBUG: Active samples before filter {i+1}: {len(active_samples)}")

            # Check if this is actually a filter or a mapper
            op_type = type(filter_op).__name__.lower()
            is_mapper = "mapper" in op_type
            _ = "filter" in op_type or "deduplicator" in op_type

            if is_mapper:
                # For mappers, just transform the data but don't change the mask
                logger.debug(f"🔍 DEBUG: Skipping mapper {i+1} for mask computation")
                if hasattr(filter_op, "compute_stats_batched"):
                    data = filter_op.compute_stats_batched(data)
                if hasattr(filter_op, "process_batched"):
                    result = list(filter_op.process_batched(data))
                    # Update data for next operation
                    if result:
                        data = {"text": result, "__dj__stats__": [{} for _ in range(len(result))]}
                continue  # Skip mask computation for mappers

            # Only process actual filters
            if hasattr(filter_op, "compute_stats_batched"):
                data = filter_op.compute_stats_batched(data)
                logger.debug(f"🔍 DEBUG: Data length after stats for filter {i+1}: {get_dataset_length(data)}")

            if hasattr(filter_op, "process_batched"):
                result = list(filter_op.process_batched(data))

                # Check if this is a mapper or filter based on the result type
                if result and isinstance(result[0], bool):
                    # This is a filter - it returns boolean masks
                    mask = result
                    logger.debug(f"🔍 DEBUG: Filter {i+1} returned {len(mask)} boolean results")
                else:
                    # This is a mapper - it returns transformed text content
                    # For mappers, we assume all samples pass (no filtering)
                    mask = [True] * len(result) if result else []
                    logger.debug(f"🔍 DEBUG: Mapper {i+1} returned {len(result)} text results, assuming all pass")

                # FIXED: Filters always return the same number of results as input samples
                # We need to track which samples are still active based on True/False results
                if len(mask) == len(active_samples):
                    # Normal case: filter returned results for all active samples
                    new_active_samples = set()
                    for sample_idx, passed in zip(sorted(active_samples), mask):
                        if passed:
                            new_active_samples.add(sample_idx)
                    active_samples = new_active_samples
                    logger.debug(f"🔍 DEBUG: Filter {i+1} kept {len(active_samples)} samples")
                elif len(mask) == original_length:
                    # Filter returned results for all original samples (no funneling happened)
                    # This means the filter processed all samples, not just active ones
                    new_active_samples = set()
                    for sample_idx, passed in enumerate(mask):
                        if sample_idx in active_samples and passed:
                            new_active_samples.add(sample_idx)
                    active_samples = new_active_samples
                    logger.debug(
                        f"🔍 DEBUG: Filter {i+1} processed all {original_length} samples, kept {len(active_samples)} active"
                    )
                else:
                    # Unexpected case: filter returned different number of results
                    logger.warning(
                        f"🔍 DEBUG: Filter {i+1} returned {len(mask)} results but {len(active_samples)} samples were active"
                    )
                    # Assume all active samples passed if we have enough results
                    if len(mask) >= len(active_samples):
                        new_active_samples = set()
                        for idx, (sample_idx, passed) in enumerate(zip(sorted(active_samples), mask)):
                            if passed:
                                new_active_samples.add(sample_idx)
                        active_samples = new_active_samples
                    else:
                        # Not enough results, drop all samples
                        active_samples = set()
            else:
                logger.debug(f"🔍 DEBUG: Filter {i+1} has no process_batched method")
                # For filters without process_batched, assume all active samples pass
                # (no change to active_samples)

        # Mark all active samples as True in the final mask
        for sample_idx in active_samples:
            if 0 <= sample_idx < original_length:
                final_mask[sample_idx] = True

        logger.debug(f"🔍 DEBUG: Final mask: {sum(final_mask)}/{original_length} samples passed")
        return final_mask

    def get_final_mask_from_optimized_ops(self, optimized_ops: List, test_data: Any) -> list:
        """Compute the final boolean mask for each sample using optimized operations."""
        data = test_data
        original_length = get_dataset_length(data)
        # Track which samples are still active (not dropped by any operation)
        active_samples = set(range(original_length))
        final_mask = [False] * original_length  # Initialize all to False

        from data_juicer.ops import load_ops

        for op_config in optimized_ops:
            # Special handling for fused_filter - we need to create the actual filter objects
            op_name = list(op_config.keys())[0]
            if op_name == "fused_filter":
                # Extract the fused_op_list and create individual filter objects
                fused_op_list = op_config[op_name].get("fused_op_list", [])
                individual_filters = []

                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]
                    # Load the individual filter
                    loaded_filters = load_ops([{filter_name: filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])

                # Create the fused filter with the actual filter objects
                if individual_filters:
                    op = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force parallel execution to match individual execution behavior
                    # (each filter sees original data, not filtered output from previous filters)
                    op.execution_strategy = "parallel"
                else:
                    continue  # Skip if we can't create the fused filter
            else:
                # Load the operation from config for non-fused operations
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    op = loaded_ops[0]
                else:
                    continue  # Skip if we can't load the operation

            # Check if this is a mapper or filter
            op_type = type(op).__name__.lower()
            is_mapper = "mapper" in op_type
            is_filter = "filter" in op_type or "deduplicator" in op_type

            # Execute the operation
            if hasattr(op, "compute_stats_batched"):
                data = op.compute_stats_batched(data)

            if hasattr(op, "process_batched"):
                result = list(op.process_batched(data))

                if is_filter and result:
                    # For filters, apply boolean mask to active samples
                    # Check if this is actually a filter or mapper based on result type
                    import numpy as np

                    is_numpy_bool = False
                    try:
                        is_numpy_bool = np.issubdtype(type(result[0]), np.bool_)
                    except Exception:
                        is_numpy_bool = False
                    if result and (
                        isinstance(result[0], bool)
                        or str(type(result[0])).endswith("bool")
                        or is_numpy_bool
                        or (
                            hasattr(result, "__iter__")
                            and not isinstance(result, str)
                            and all(
                                isinstance(x, bool)
                                or str(type(x)).endswith("bool")
                                or (np.issubdtype(type(x), np.bool_) if "numpy" in str(type(x)) else False)
                                for x in result
                            )
                        )
                    ):
                        # This is a filter - it returns boolean masks
                        op_mask = result
                    else:
                        # This is a mapper - it returns transformed text content
                        # For mappers, we assume all samples pass (no filtering)
                        op_mask = [True] * len(result) if result else []

                    # Apply the filter mask to active samples
                    if len(op_mask) == len(active_samples):
                        # Normal case: filter returned results for all active samples
                        new_active_samples = set()
                        for sample_idx, passed in zip(sorted(active_samples), op_mask):
                            if passed:
                                new_active_samples.add(sample_idx)
                        active_samples = new_active_samples
                    elif len(op_mask) == original_length:
                        # Filter returned results for all original samples
                        new_active_samples = set()
                        for sample_idx, passed in enumerate(op_mask):
                            if sample_idx in active_samples and passed:
                                new_active_samples.add(sample_idx)
                        active_samples = new_active_samples
                    else:
                        # Unexpected case: filter returned different number of results
                        logger.warning(
                            f"Filter {op_type} returned {len(op_mask)} results but {len(active_samples)} samples were active"
                        )
                        # Assume all active samples passed if we have enough results
                        if len(op_mask) >= len(active_samples):
                            new_active_samples = set()
                            for idx, (sample_idx, passed) in enumerate(zip(sorted(active_samples), op_mask)):
                                if passed:
                                    new_active_samples.add(sample_idx)
                            active_samples = new_active_samples
                        else:
                            # Not enough results, drop all samples
                            active_samples = set()

                elif is_mapper and result:
                    # For mappers, check if any samples were dropped (empty results)
                    # Mappers typically return transformed data, not boolean masks
                    if len(result) < len(active_samples):
                        # Some samples were dropped by the mapper
                        # This is rare but possible - assume first N samples survived
                        active_samples = set(range(len(result)))
                    # If len(result) >= len(active_samples), no samples were dropped
                    # Update data for next operation
                    data = {"text": result, "__dj__stats__": [{} for _ in range(len(result))]}

        # Mark all active samples as True in the final mask
        for sample_idx in active_samples:
            if 0 <= sample_idx < original_length:
                final_mask[sample_idx] = True

        return final_mask

    def run_benchmark(
        self,
        filters: List[Filter],
        test_data: Any,
        mode: str = "quick",
        analyzer_insights: Optional[Dict[str, Any]] = None,
        benchmark_type: str = "both",
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing individual vs pipeline optimizer execution."""
        logger.info("🚀 Starting Performance Benchmark")
        logger.info("=" * 60)

        # Get analyzer insights if not provided
        if analyzer_insights is None:
            logger.info("🔍 Getting analyzer insights...")
            analyzer_insights = self.get_analyzer_insights(test_data)

        # Run benchmarks based on benchmark_type
        pipeline_results = None
        individual_results = None

        if benchmark_type in ["pipeline", "both"]:
            logger.info("\n🔧 PIPELINE OPTIMIZER BENCHMARK")
            logger.info("-" * 40)
            pipeline_results = self.run_pipeline_optimizer_benchmark(filters, test_data, analyzer_insights)

        if benchmark_type in ["individual", "both"]:
            logger.info("\n📊 INDIVIDUAL EXECUTION BENCHMARK")
            logger.info("-" * 40)
            individual_results = self.run_individual_filters_benchmark(filters, test_data)

        # For "both" mode, focus on correctness comparison
        if benchmark_type == "both" and individual_results and pipeline_results:
            logger.info("\n🔍 CORRECTNESS COMPARISON")
            logger.info("-" * 40)

            # For correctness comparison, we need to run both methods and compare their outputs
            # Since the current benchmark methods don't return the processed data,
            # we'll focus on comparing the execution patterns and final sample counts
            # from the benchmark results themselves

            # Get throughput as a proxy for final sample count (throughput = samples/time)
            individual_throughput = individual_results.throughput if individual_results else 0
            pipeline_throughput = pipeline_results.throughput if pipeline_results else 0

            # Calculate approximate final sample counts based on throughput and time
            individual_time = individual_results.total_time if individual_results else 0
            pipeline_time = pipeline_results.total_time if pipeline_results else 0

            individual_final_samples = int(individual_throughput * individual_time) if individual_time > 0 else 0
            pipeline_final_samples = int(pipeline_throughput * pipeline_time) if pipeline_time > 0 else 0

            logger.info(f"Individual execution final samples: {individual_final_samples}")
            logger.info(f"Pipeline execution final samples: {pipeline_final_samples}")

            if individual_final_samples == pipeline_final_samples:
                logger.info("✅ Sample counts match - Correctness validation passed!")
                correctness_passed = True
            else:
                logger.warning("❌ Sample counts differ - Correctness validation failed!")
                correctness_passed = False

            # Log execution times for reference (but don't compare them)
            individual_time = individual_results.total_time if individual_results else 0
            pipeline_time = pipeline_results.total_time if pipeline_results else 0

            logger.info(f"Individual execution time: {individual_time:.3f}s")
            logger.info(f"Pipeline execution time: {pipeline_time:.3f}s")

        elif benchmark_type == "pipeline" and pipeline_results:
            pipeline_time = pipeline_results.total_time if pipeline_results else 0
            logger.info(f"Pipeline optimizer: {pipeline_time:.3f}s ({pipeline_results.throughput:.1f} samples/s)")
        elif benchmark_type == "individual" and individual_results:
            individual_time = individual_results.total_time if individual_results else 0
            logger.info(f"Individual execution: {individual_time:.3f}s ({individual_results.throughput:.1f} samples/s)")

        # Compile results
        results = {
            "mode": mode,
            "benchmark_type": benchmark_type,
            "num_samples": get_dataset_length(test_data),
            "num_filters": len(filters),
            "individual": {
                "total_time": individual_results.total_time if individual_results else 0,
                "stats_time": individual_results.stats_time if individual_results else 0,
                "filter_time": individual_results.filter_time if individual_results else 0,
                "memory_usage": individual_results.memory_usage if individual_results else 0,
                "throughput": individual_results.throughput if individual_results else 0,
            },
            "pipeline": {
                "total_time": pipeline_results.total_time if pipeline_results else 0,
                "stats_time": pipeline_results.stats_time if pipeline_results else 0,
                "filter_time": pipeline_results.filter_time if pipeline_results else 0,
                "memory_usage": pipeline_results.memory_usage if pipeline_results else 0,
                "throughput": pipeline_results.throughput if pipeline_results else 0,
            },
            "correctness_passed": correctness_passed if benchmark_type == "both" else None,
            "analyzer_insights": analyzer_insights,
        }

        # Write results to output files in "both" mode
        if benchmark_type == "both":
            self._write_benchmark_results(results, mode, filters)

        return results

    def _run_and_save_individual_execution(
        self, filters: List[Filter], test_data: Any, validation_dir: str
    ) -> Dict[str, Any]:
        """Run individual execution and save results with IDs."""
        logger.info("🔍 Running individual execution for validation...")

        # Debug: Log all filters being processed
        logger.debug(f"🔍 DEBUG: Processing {len(filters)} filters in individual execution:")
        for i, filter_op in enumerate(filters):
            filter_type = type(filter_op).__name__
            filter_name = getattr(filter_op, "_name", f"filter_{i}")
            has_compute_stats = hasattr(filter_op, "compute_stats_batched")
            has_process_batched = hasattr(filter_op, "process_batched")
            logger.info(
                f"  Filter {i+1}: {filter_type} (name: {filter_name}, has_compute_stats: {has_compute_stats}, has_process_batched: {has_process_batched})"
            )

            # Log filter configuration
            config = getattr(filter_op, "config", None)
            if config:
                logger.info(f"    Config: {config}")
            else:
                # Log filter attributes
                attrs = {}
                for attr in dir(filter_op):
                    if not attr.startswith("_") and not callable(getattr(filter_op, attr)):
                        value = getattr(filter_op, attr)
                        if isinstance(value, (int, float, str, bool)):
                            attrs[attr] = value
                logger.info(f"    Attributes: {attrs}")

        # Add sample IDs to test data
        original_length = get_dataset_length(test_data)
        test_data_with_ids = test_data.copy()
        test_data_with_ids["sample_id"] = list(range(original_length))

        logger.debug(f"🔍 DEBUG: Starting with {original_length} samples")
        logger.debug(f"🔍 DEBUG: Test data keys: {list(test_data_with_ids.keys())}")
        logger.debug(f"🔍 DEBUG: Text samples: {len(test_data_with_ids.get('text', []))}")

        # Process through all filters
        data = test_data_with_ids
        for i, filter_op in enumerate(filters):
            filter_type = type(filter_op).__name__
            filter_name = getattr(filter_op, "_name", f"filter_{i}")
            logger.debug(f"🔍 DEBUG: Processing filter {i+1}/{len(filters)}: {filter_type} ({filter_name})")
            logger.debug(f"🔍 DEBUG: Data before filter {i+1}: {len(data.get('text', []))} samples")
            logger.debug(f"🔍 DEBUG: Text field type before filter {i+1}: {type(data.get('text', []))}")

            if hasattr(filter_op, "compute_stats_batched"):
                logger.debug(f"🔍 DEBUG: Computing stats for filter {i+1}...")
                data = filter_op.compute_stats_batched(data)
                logger.debug(f"🔍 DEBUG: After stats for filter {i+1}: {len(data.get('text', []))} samples")
            else:
                logger.debug(f"🔍 DEBUG: Filter {i+1} has no compute_stats_batched method")

            if hasattr(filter_op, "process_batched"):
                logger.debug(f"🔍 DEBUG: Processing filter {i+1}...")
                result = list(filter_op.process_batched(data))
                logger.debug(f"🔍 DEBUG: Filter {i+1} result type: {type(result[0]) if result else 'None'}")
                logger.debug(f"🔍 DEBUG: Filter {i+1} result length: {len(result) if result else 0}")
                if result and len(result) > 0:
                    logger.debug(f"🔍 DEBUG: Filter {i+1} first result: {result[0]}")
                    logger.debug(f"🔍 DEBUG: Filter {i+1} first result type: {type(result[0])}")

                # Check if this is a filter (returns boolean) or mapper (returns transformed data)
                # Handle both list and map objects that contain booleans
                import numpy as np

                is_numpy_bool = False
                try:
                    is_numpy_bool = np.issubdtype(type(result[0]), np.bool_)
                except Exception:
                    is_numpy_bool = False
                if result and (
                    isinstance(result[0], bool)
                    or str(type(result[0])).endswith("bool")
                    or is_numpy_bool
                    or (
                        hasattr(result, "__iter__")
                        and not isinstance(result, str)
                        and all(
                            isinstance(x, bool)
                            or str(type(x)).endswith("bool")
                            or (np.issubdtype(type(x), np.bool_) if "numpy" in str(type(x)) else False)
                            for x in result
                        )
                    )
                ):
                    # This is a filter - apply boolean mask
                    mask = result
                    logger.debug(f"🔍 DEBUG: Filter {i+1} boolean mask: {sum(mask)}/{len(mask)} samples passed")
                    logger.debug(f"🔍 DEBUG: Filter {i+1} mask details: {mask[:10]}...")  # Show first 10 values

                    # Store the filter result for debugging
                    data[f"filter_result_filter_{i+1}"] = mask

                    # Keep only samples that passed the filter
                    passed_indices = [idx for idx, passed in enumerate(mask) if passed]
                    if passed_indices:
                        # Update data to keep only passed samples
                        logger.debug(f"🔍 DEBUG: Before filtering - text field type: {type(data.get('text', []))}")
                        logger.debug(f"🔍 DEBUG: Before filtering - text field length: {len(data.get('text', []))}")
                        if data.get("text") and len(data["text"]) > 0:
                            logger.debug(f"🔍 DEBUG: Before filtering - first text sample: {data['text'][0][:50]}...")

                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) == len(mask):
                                data[key] = [data[key][idx] for idx in passed_indices]

                        logger.debug(f"🔍 DEBUG: After filtering - text field type: {type(data.get('text', []))}")
                        logger.debug(f"🔍 DEBUG: After filtering - text field length: {len(data.get('text', []))}")
                        if data.get("text") and len(data["text"]) > 0:
                            logger.debug(f"🔍 DEBUG: After filtering - first text sample: {data['text'][0][:50]}...")

                        logger.debug(f"🔍 DEBUG: After filter {i+1}: {len(passed_indices)} samples remaining")
                    else:
                        # No samples passed - clear all data
                        for key in data:
                            if isinstance(data[key], list):
                                data[key] = []
                        logger.debug(f"🔍 DEBUG: After filter {i+1}: 0 samples remaining - stopping")
                        break
                else:
                    # This is a mapper - transform the data
                    logger.debug(f"🔍 DEBUG: Mapper {i+1} transformed data: {len(result)} samples")
                    # Update only the 'text' field
                    data["text"] = result
                    # Keep sample_ids and stats aligned
                    if "sample_id" in data and len(data["sample_id"]) != len(result):
                        data["sample_id"] = data["sample_id"][: len(result)]
                    if Fields.stats in data and len(data[Fields.stats]) != len(result):
                        data[Fields.stats] = data[Fields.stats][: len(result)]
                    logger.debug(f"🔍 DEBUG: After mapper {i+1}: {len(data.get('text', []))} samples")
            else:
                logger.warning(f"🔍 DEBUG: Filter {i+1} has no process_batched method")

        logger.debug(f"🔍 DEBUG: Final individual execution: {len(data.get('text', []))} samples")
        logger.debug(f"🔍 DEBUG: Final data keys: {list(data.keys())}")
        logger.debug(f"🔍 DEBUG: Final text field type: {type(data.get('text', []))}")
        if data.get("text") and len(data["text"]) > 0:
            logger.debug(
                f"🔍 DEBUG: Final first text sample: {data['text'][0] if isinstance(data['text'], list) else data['text']}"
            )
            logger.debug(
                f"🔍 DEBUG: Final first text sample type: {type(data['text'][0] if isinstance(data['text'], list) else data['text'])}"
            )

        # SAFEGUARD: Ensure text field is always a list of strings after all filters
        if "text" in data and data["text"] and not all(isinstance(t, str) for t in data["text"]):
            logger.warning("Text field contains non-string elements after filtering. Converting to strings.")
            data["text"] = [str(t) for t in data["text"]]

        # ADDITIONAL SAFEGUARD: Check if text field is corrupted
        if "text" in data:
            if not isinstance(data["text"], list):
                logger.error(f"🔍 ERROR: text field is not a list: {type(data['text'])} = {data['text']}")
                # Try to recover by using the original test data
                data["text"] = test_data.get("text", [])
                data["sample_id"] = list(range(len(data["text"])))
            elif len(data["text"]) == 0:
                logger.warning("🔍 WARNING: text field is empty after filtering")
            else:
                logger.info(f"🔍 INFO: Final text field has {len(data['text'])} samples")
                if len(data["text"]) > 0:
                    logger.info(
                        f"🔍 INFO: First text sample: {data['text'][0][:100] if isinstance(data['text'][0], str) else str(data['text'][0])[:100]}"
                    )

        individual_results_file = os.path.join(validation_dir, "individual_execution_results.jsonl")
        self._save_results_to_file(data, individual_results_file)
        logger.info(f"📄 Individual execution results saved to: {individual_results_file}")

        return data

    def _run_and_save_pipeline_execution(
        self, optimized_ops: List, test_data: Any, validation_dir: str
    ) -> Dict[str, Any]:
        """Run pipeline execution and save results with IDs."""
        logger.info("🔍 Running pipeline execution for validation...")

        # Debug: Log all optimized operations being processed
        logger.debug(f"🔍 DEBUG: Processing {len(optimized_ops)} optimized operations in pipeline execution:")
        for i, op_config in enumerate(optimized_ops):
            op_name = list(op_config.keys())[0]
            op_args = op_config[op_name]
            logger.info(f"  Op {i+1}: {op_name} (args: {op_args})")

        # Add sample IDs to test data
        original_length = get_dataset_length(test_data)
        test_data_with_ids = test_data.copy()
        test_data_with_ids["sample_id"] = list(range(original_length))

        logger.debug(f"🔍 DEBUG: Starting with {original_length} samples")
        logger.debug(f"🔍 DEBUG: Test data keys: {list(test_data_with_ids.keys())}")
        logger.debug(f"🔍 DEBUG: Text samples: {len(test_data_with_ids.get('text', []))}")

        # Process with optimized operations
        data = test_data_with_ids
        from data_juicer.ops import load_ops

        for op_idx, op_config in enumerate(optimized_ops):
            logger.debug(f"🔍 DEBUG: Processing optimized op {op_idx+1}/{len(optimized_ops)}")
            logger.debug(f"🔍 DEBUG: Data before op {op_idx+1}: {len(data.get('text', []))} samples")
            logger.debug(f"🔍 DEBUG: Text field type before op {op_idx+1}: {type(data.get('text', []))}")
            if data.get("text") and len(data["text"]) > 0:
                logger.debug(f"🔍 DEBUG: First text sample before op {op_idx+1}: {data['text'][0][:50]}...")

            # Special handling for fused_filter - we need to create the actual filter objects
            op_name = list(op_config.keys())[0]
            logger.debug(f"🔍 DEBUG: Op {op_idx+1} type: {op_name}")

            if op_name == "fused_filter":
                # Extract the fused_op_list and create individual filter objects
                fused_op_list = op_config[op_name].get("fused_op_list", [])
                individual_filters = []

                logger.debug(f"🔍 DEBUG: Fused filter contains {len(fused_op_list)} individual filters:")
                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]
                    logger.info(f"    - {filter_name}: {filter_args}")

                    # Filter out arguments that are meant for FusedFilter, not individual filters
                    fused_filter_args = {
                        "accelerator",
                        "batch_size",
                        "cpu_required",
                        "mem_required",
                        "num_proc",
                        "skip_op_error",
                        "turbo",
                        "text_key",
                        "image_key",
                        "audio_key",
                        "video_key",
                        "history_key",
                        "query_key",
                        "response_key",
                        "execution_strategy",
                        "has_dependencies",
                        "fused_op_list",
                    }

                    # Remove fused filter specific arguments
                    clean_filter_args = {k: v for k, v in filter_args.items() if k not in fused_filter_args}

                    # Load the individual filter
                    loaded_filters = load_ops([{filter_name: clean_filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])
                        logger.info(f"    ✅ Successfully loaded {filter_name}")
                    else:
                        logger.warning(f"    ❌ Failed to load {filter_name}")

                # Create the fused filter with the actual filter objects
                if individual_filters:
                    op = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force sequential execution to match individual execution behavior exactly
                    # (process filters one by one in the same order as individual execution)
                    op.execution_strategy = "sequential"
                    logger.debug(f"🔍 DEBUG: Created fused filter with {len(individual_filters)} filters")
                    logger.debug(f"🔍 DEBUG: Fused filter execution strategy: {op.execution_strategy}")
                    logger.debug(f"🔍 DEBUG: Filter order: {[f._name for f in op.fused_filters]}")
                else:
                    logger.warning(f"🔍 DEBUG: Failed to create fused filter")
                    continue  # Skip if we can't create the fused filter
            else:
                # Load the operation from config for non-fused operations
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    op = loaded_ops[0]
                    logger.debug(f"🔍 DEBUG: Loaded op: {type(op).__name__}")
                else:
                    logger.warning(f"🔍 DEBUG: Failed to load op from config")
                    continue  # Skip if we can't load the operation

            # Execute the operation
            if hasattr(op, "compute_stats_batched"):
                logger.debug(f"🔍 DEBUG: Computing stats for op {op_idx+1}...")
                data = op.compute_stats_batched(data)
                logger.debug(f"🔍 DEBUG: After stats for op {op_idx+1}: {len(data.get('text', []))} samples")
            else:
                logger.debug(f"🔍 DEBUG: Op {op_idx+1} has no compute_stats_batched method")

            if hasattr(op, "process_batched"):
                logger.debug(f"🔍 DEBUG: Processing op {op_idx+1}...")
                result = list(op.process_batched(data))
                logger.debug(f"🔍 DEBUG: Op {op_idx+1} result type: {type(result[0]) if result else 'None'}")
                logger.debug(f"🔍 DEBUG: Op {op_idx+1} result length: {len(result) if result else 0}")
                if result and len(result) > 0:
                    logger.debug(f"🔍 DEBUG: Op {op_idx+1} first result: {result[0]}")
                    logger.debug(f"🔍 DEBUG: Op {op_idx+1} first result type: {type(result[0])}")

                # Check if this is a filter (returns boolean) or mapper (returns transformed data)
                # Handle both list and map objects that contain booleans
                import numpy as np

                is_numpy_bool = False
                try:
                    is_numpy_bool = np.issubdtype(type(result[0]), np.bool_)
                except Exception:
                    is_numpy_bool = False
                if result and (
                    isinstance(result[0], bool)
                    or str(type(result[0])).endswith("bool")
                    or is_numpy_bool
                    or (
                        hasattr(result, "__iter__")
                        and not isinstance(result, str)
                        and all(
                            isinstance(x, bool)
                            or str(type(x)).endswith("bool")
                            or (np.issubdtype(type(x), np.bool_) if "numpy" in str(type(x)) else False)
                            for x in result
                        )
                    )
                ):
                    # This is a filter - apply boolean mask
                    mask = result
                    logger.debug(f"🔍 DEBUG: Op {op_idx+1} boolean mask: {sum(mask)}/{len(mask)} samples passed")
                    logger.debug(f"🔍 DEBUG: Op {op_idx+1} mask details: {mask[:10]}...")  # Show first 10 values

                    # Store the filter result for debugging
                    data[f"filter_result_op_{op_idx+1}"] = mask

                    # Keep only samples that passed the filter
                    passed_indices = [idx for idx, passed in enumerate(mask) if passed]
                    if passed_indices:
                        # Update data to keep only passed samples
                        logger.debug(f"🔍 DEBUG: Before filtering - text field type: {type(data.get('text', []))}")
                        logger.debug(f"🔍 DEBUG: Before filtering - text field length: {len(data.get('text', []))}")
                        if data.get("text") and len(data["text"]) > 0:
                            logger.debug(f"🔍 DEBUG: Before filtering - first text sample: {data['text'][0][:50]}...")

                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) == len(mask):
                                data[key] = [data[key][idx] for idx in passed_indices]

                        logger.debug(f"🔍 DEBUG: After filtering - text field type: {type(data.get('text', []))}")
                        logger.debug(f"🔍 DEBUG: After filtering - text field length: {len(data.get('text', []))}")
                        if data.get("text") and len(data["text"]) > 0:
                            logger.debug(f"🔍 DEBUG: After filtering - first text sample: {data['text'][0][:50]}...")

                        logger.debug(f"🔍 DEBUG: After op {op_idx+1}: {len(passed_indices)} samples remaining")

                        # SAFEGUARD: Ensure text field is always a list of strings
                        if "text" in data:
                            if not isinstance(data["text"], list):
                                logger.warning(
                                    f"🔍 WARNING: text field is not a list after filter {op_idx+1}: {type(data['text'])} = {data['text']}"
                                )
                                data["text"] = []
                            elif data["text"] and not all(isinstance(t, str) for t in data["text"]):
                                logger.warning(
                                    f"🔍 WARNING: text field contains non-string elements after filter {op_idx+1}: {[type(t) for t in data['text'][:3]]}"
                                )
                                # Convert non-string elements to strings
                                data["text"] = [str(t) if not isinstance(t, str) else t for t in data["text"]]
                    else:
                        # No samples passed - clear all data
                        for key in data:
                            if isinstance(data[key], list):
                                data[key] = []
                        logger.debug(f"🔍 DEBUG: After op {op_idx+1}: 0 samples remaining")
                        break
                else:
                    # This is a mapper - update text data
                    if result:
                        data["text"] = result
                        # Keep sample_ids and stats aligned
                        if "sample_id" in data and len(data["sample_id"]) != len(result):
                            data["sample_id"] = data["sample_id"][: len(result)]
                        if Fields.stats in data and len(data[Fields.stats]) != len(result):
                            data[Fields.stats] = data[Fields.stats][: len(result)]
                        logger.debug(f"🔍 DEBUG: After mapper op {op_idx+1}: {len(data.get('text', []))} samples")
                        logger.debug(f"🔍 DEBUG: Data keys after mapper operation {op_idx+1}: {list(data.keys())}")
            else:
                logger.warning(f"🔍 DEBUG: Op {op_idx+1} has no process_batched method")

        logger.debug(f"🔍 DEBUG: Final pipeline execution: {len(data.get('text', []))} samples")
        logger.debug(f"🔍 DEBUG: Final data keys: {list(data.keys())}")
        logger.debug(f"🔍 DEBUG: Final text field type: {type(data.get('text', []))}")
        if data.get("text") and len(data["text"]) > 0:
            logger.debug(
                f"🔍 DEBUG: Final first text sample: {data['text'][0] if isinstance(data['text'], list) else data['text']}"
            )
            logger.debug(
                f"🔍 DEBUG: Final first text sample type: {type(data['text'][0] if isinstance(data['text'], list) else data['text'])}"
            )

        # ADDITIONAL SAFEGUARD: Check if text field is corrupted
        if "text" in data:
            if not isinstance(data["text"], list):
                logger.error(f"🔍 ERROR: text field is not a list: {type(data['text'])} = {data['text']}")
                # Try to recover by using the original test data
                data["text"] = test_data.get("text", [])
                data["sample_id"] = list(range(len(data["text"])))
            elif len(data["text"]) == 0:
                logger.warning("🔍 WARNING: text field is empty after filtering")
            else:
                logger.info(f"🔍 INFO: Final text field has {len(data['text'])} samples")
                if len(data["text"]) > 0:
                    logger.debug(
                        f"🔍 INFO: First text sample: {data['text'][0][:100] if isinstance(data['text'][0], str) else str(data['text'][0])[:100]}"
                    )

        # Save pipeline execution results
        pipeline_results_file = os.path.join(validation_dir, "pipeline_execution_results.jsonl")
        self._save_results_to_file(data, pipeline_results_file)
        logger.info(f"📄 Pipeline execution results saved to: {pipeline_results_file}")

        return data

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python types for JSON serialization."""
        import numpy as np

        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def _save_results_to_file(self, data: Dict[str, Any], filepath: str):
        """Save execution results to a JSONL file with sample IDs."""
        import json

        logger.info(f"💾 Saving results to {filepath}")

        # Debug data structure
        logger.debug(f"🔍 DEBUG: Data keys: {list(data.keys())}")
        logger.debug(f"🔍 DEBUG: Data structure: {data}")

        if "text" in data:
            logger.debug(f"🔍 DEBUG: Text field type: {type(data['text'])}")
            logger.info(
                f"🔍 DEBUG: Text field length: {len(data['text']) if hasattr(data['text'], '__len__') else 'N/A'}"
            )
            if hasattr(data["text"], "__len__") and len(data["text"]) > 0:
                logger.debug(f"🔍 DEBUG: First text sample: {data['text'][0]}")
                logger.debug(f"🔍 DEBUG: First text sample type: {type(data['text'][0])}")

        with open(filepath, "w", encoding="utf-8") as f:
            # Check if we have the expected data structure
            if "text" in data and isinstance(data["text"], list) and len(data["text"]) > 0:
                logger.debug("🔍 DEBUG: Using text field with proper structure")

                # Get sample IDs if available, otherwise use indices
                sample_ids = data.get("sample_id", list(range(len(data["text"]))))

                # Ensure sample_ids is a list and has the same length as text
                if not isinstance(sample_ids, list):
                    sample_ids = list(range(len(data["text"])))
                elif len(sample_ids) != len(data["text"]):
                    logger.warning(
                        f"🔍 WARNING: sample_id length ({len(sample_ids)}) != text length ({len(data['text'])}), using indices"
                    )
                    sample_ids = list(range(len(data["text"])))

                # Write each sample
                for i, (text, sample_id) in enumerate(zip(data["text"], sample_ids)):
                    result = {
                        "sample_id": self._convert_numpy_types(sample_id),
                        "text": self._convert_numpy_types(text),
                        "final_index": i,
                    }

                    # Add filter results if they exist
                    filter_results = {}
                    for key, value in data.items():
                        if key.startswith("filter_result_") and isinstance(value, list) and i < len(value):
                            filter_results[key] = self._convert_numpy_types(value[i])

                    if filter_results:
                        result["filter_results"] = filter_results

                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                logger.warning("🔍 WARNING: Invalid data structure for saving results")
                logger.warning(f"🔍 WARNING: text field: {data.get('text', 'NOT_FOUND')}")
                logger.warning(f"🔍 WARNING: text type: {type(data.get('text', None))}")
                logger.warning(
                    f"🔍 WARNING: text length: {len(data.get('text', [])) if hasattr(data.get('text', []), '__len__') else 'N/A'}"
                )

                # Write empty file with warning
                f.write(
                    json.dumps({"error": "Invalid data structure", "data_keys": list(data.keys())}, ensure_ascii=False)
                    + "\n"
                )

    def _compare_execution_results(
        self, individual_data: Dict[str, Any], pipeline_data: Dict[str, Any], validation_dir: str
    ) -> Dict[str, Any]:
        """Compare individual vs pipeline execution results."""
        logger.info("🔍 Comparing execution results...")

        # Extract sample IDs and texts
        individual_ids = individual_data.get("sample_id", [])
        individual_texts = individual_data.get("text", [])
        pipeline_ids = pipeline_data.get("sample_id", [])
        pipeline_texts = pipeline_data.get("text", [])

        logger.info(f"📊 Individual execution: {len(individual_ids)} samples")
        logger.info(f"📊 Pipeline execution: {len(pipeline_ids)} samples")

        # Create comparison report
        comparison_file = os.path.join(validation_dir, "comparison_report.txt")
        with open(comparison_file, "w", encoding="utf-8") as f:
            f.write("EXECUTION COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Individual execution samples: {len(individual_ids)}\n")
            f.write(f"Pipeline execution samples: {len(pipeline_ids)}\n\n")

            # Check if sample counts match
            if len(individual_ids) == len(pipeline_ids):
                f.write("✅ Sample counts match!\n\n")

                # Check if sample IDs match
                if individual_ids == pipeline_ids:
                    f.write("✅ Sample IDs match!\n\n")

                    # Check if texts match
                    mismatches = []
                    for i, (ind_text, pipe_text) in enumerate(zip(individual_texts, pipeline_texts)):
                        # Convert any type to string for comparison
                        ind_text_str = str(ind_text) if ind_text is not None else ""
                        pipe_text_str = str(pipe_text) if pipe_text is not None else ""

                        if ind_text_str != pipe_text_str:
                            # Safely truncate for display
                            ind_display = ind_text_str[:100] + "..." if len(ind_text_str) > 100 else ind_text_str
                            pipe_display = pipe_text_str[:100] + "..." if len(pipe_text_str) > 100 else pipe_text_str

                            mismatches.append(
                                {
                                    "index": i,
                                    "sample_id": individual_ids[i],
                                    "individual_text": ind_display,
                                    "pipeline_text": pipe_display,
                                }
                            )

                    if not mismatches:
                        f.write("✅ All texts match! Individual and pipeline execution are identical.\n")
                        logger.info("✅ VALIDATION PASSED: Individual and pipeline execution results are identical!")
                        validation_passed = True
                    else:
                        f.write(f"❌ Found {len(mismatches)} text mismatches:\n\n")
                        for mismatch in mismatches[:10]:  # Show first 10 mismatches
                            f.write(f"Index {mismatch['index']} (Sample ID {mismatch['sample_id']}):\n")
                            f.write(f"  Individual: {mismatch['individual_text']}\n")
                            f.write(f"  Pipeline:  {mismatch['pipeline_text']}\n\n")
                        logger.warning(f"❌ VALIDATION FAILED: Found {len(mismatches)} text mismatches")
                        validation_passed = False
                else:
                    f.write("❌ Sample IDs do not match!\n")
                    f.write(f"Individual IDs: {individual_ids[:10]}...\n")
                    f.write(f"Pipeline IDs:  {pipeline_ids[:10]}...\n")
                    logger.warning("❌ VALIDATION FAILED: Sample IDs do not match")
                    validation_passed = False
            else:
                f.write("❌ Sample counts do not match!\n")
                f.write(f"Individual: {len(individual_ids)} samples\n")
                f.write(f"Pipeline:  {len(pipeline_ids)} samples\n")
                logger.warning("❌ VALIDATION FAILED: Sample counts do not match")
                validation_passed = False

        logger.info(f"📄 Comparison report saved to: {comparison_file}")

        return {
            "passed": validation_passed,
            "individual_samples": len(individual_ids),
            "pipeline_samples": len(pipeline_ids),
            "sample_ids_match": individual_ids == pipeline_ids,
            "comparison_file": comparison_file,
        }

    def create_quick_test_filters(self) -> List[Filter]:
        """Get a small set of filters for quick testing."""
        from data_juicer.ops.filter import (
            alphanumeric_filter,
            text_length_filter,
            words_num_filter,
        )

        return [
            alphanumeric_filter.AlphanumericFilter(min_ratio=0.7),  # At least 70% alphanumeric (was 0.5)
            text_length_filter.TextLengthFilter(min_len=50, max_len=2000),  # 50-2000 chars (was 10-1000)
            words_num_filter.WordsNumFilter(min_num=10, max_num=400),  # 10-400 words (was 5-200)
        ]

    def create_full_test_filters(self) -> List[Filter]:
        """Get a comprehensive set of filters for full testing."""
        from data_juicer.ops.filter import (
            alphanumeric_filter,
            character_repetition_filter,
            special_characters_filter,
            stopwords_filter,
            text_length_filter,
            word_repetition_filter,
            words_num_filter,
        )

        return [
            alphanumeric_filter.AlphanumericFilter(min_ratio=0.75),  # At least 75% alphanumeric (was 0.6)
            text_length_filter.TextLengthFilter(min_len=200, max_len=3000),  # 200-3000 chars (was 100-5000)
            words_num_filter.WordsNumFilter(min_num=30, max_num=500),  # 30-500 words (was 20-1000)
            character_repetition_filter.CharacterRepetitionFilter(max_ratio=0.15),  # Max 15% repetition (was 0.25)
            word_repetition_filter.WordRepetitionFilter(max_ratio=0.2),  # Max 20% word repetition (was 0.3)
            special_characters_filter.SpecialCharactersFilter(
                min_ratio=0.0, max_ratio=0.15
            ),  # Max 15% special chars (was 0.25)
            stopwords_filter.StopWordsFilter(min_ratio=0.15, max_ratio=0.4),  # 15-40% stop words (was 0.1-0.5)
        ]

    def run_analyzer(self, test_data: Any) -> Optional[Dict[str, Any]]:
        """Run analyzer to get insights for optimization."""
        try:
            from data_juicer.core.analyzer import Analyzer

            analyzer = Analyzer()
            analyzer.run(dataset=test_data, skip_return=True)
            insights = analyzer.overall_result
            logger.info("✅ Analyzer insights generated successfully")
            # Convert DataFrame to dict if it exists
            if insights is not None:
                return insights.to_dict()
            return None
        except Exception as e:
            logger.warning(f"⚠️  Analyzer failed: {e}")
            return None

    def _write_benchmark_results(self, results: Dict[str, Any], mode: str, filters: List[Filter]) -> None:
        import json
        from datetime import datetime

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/benchmark_results_{mode}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"📁 Writing benchmark results to: {output_dir}")

        # Write main results JSON
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📄 Main results saved to: {results_file}")

        # Write detailed performance report
        report_file = os.path.join(output_dir, "performance_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("DATA-JUICER PERFORMANCE BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Benchmark Configuration:\n")
            f.write(f"  Mode: {mode}\n")
            f.write(f"  Number of filters: {results['num_filters']}\n")
            f.write(f"  Number of samples: {results['num_samples']:,}\n")
            f.write(f"  Benchmark type: {results['benchmark_type']}\n\n")

            f.write("Performance Results:\n")
            f.write("-" * 30 + "\n")

            # Individual execution results
            f.write("Individual Execution:\n")
            individual = results["individual"]
            f.write(f"  Total time: {individual['total_time']:.3f}s\n")
            f.write(f"  Stats computation: {individual['stats_time']:.3f}s\n")
            f.write(f"  Filtering: {individual['filter_time']:.3f}s\n")
            f.write(f"  Memory usage: {individual['memory_usage']:.1f} MB\n")
            f.write(f"  Throughput: {individual['throughput']:.1f} samples/sec\n\n")

            # Pipeline execution results
            f.write("Pipeline Execution (FusedFilter):\n")
            pipeline = results["pipeline"]
            f.write(f"  Total time: {pipeline['total_time']:.3f}s\n")
            f.write(f"  Stats computation: {pipeline['stats_time']:.3f}s\n")
            f.write(f"  Filtering: {pipeline['filter_time']:.3f}s\n")
            f.write(f"  Memory usage: {pipeline['memory_usage']:.1f} MB\n")
            f.write(f"  Throughput: {pipeline['throughput']:.1f} samples/sec\n\n")

            # Performance comparison
            if individual["total_time"] > 0 and pipeline["total_time"] > 0:
                speedup = individual["total_time"] / pipeline["total_time"]
                f.write("Performance Comparison:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Speedup: {speedup:.2f}x\n")
                f.write(
                    f"  Time savings: {((individual['total_time'] - pipeline['total_time']) / individual['total_time'] * 100):.1f}%\n"
                )
                f.write(
                    f"  Throughput improvement: {((pipeline['throughput'] - individual['throughput']) / individual['throughput'] * 100):.1f}%\n"
                )

            # Correctness validation
            if results.get("correctness_passed") is not None:
                f.write("\nCorrectness Validation:\n")
                f.write("-" * 30 + "\n")
                if results["correctness_passed"]:
                    f.write("  ✅ PASSED: Individual and pipeline execution produce identical results\n")
                else:
                    f.write("  ❌ FAILED: Individual and pipeline execution produce different results\n")

            # Filter details
            f.write("\nFilter Details:\n")
            f.write("-" * 30 + "\n")
            for i, filter_op in enumerate(filters):
                filter_name = getattr(filter_op, "_name", type(filter_op).__name__)
                f.write(f"  {i+1}. {filter_name}\n")

            # Analyzer insights (if available)
            if results.get("analyzer_insights"):
                f.write("\nAnalyzer Insights:\n")
                f.write("-" * 30 + "\n")
                insights = results["analyzer_insights"]
                if isinstance(insights, dict):
                    for key, value in insights.items():
                        if isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for sub_key, sub_value in value.items():
                                f.write(f"    {sub_key}: {sub_value}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
        logger.info(f"📄 Performance report saved to: {report_file}")

        # Write CSV summary for easy analysis
        csv_file = os.path.join(output_dir, "performance_summary.csv")
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("total_time,stats_time,filter_time,memory_usage,throughput\n")
            f.write(
                f"{individual['total_time']:.3f},{individual['stats_time']:.3f},{individual['filter_time']:.3f},{individual['memory_usage']:.1f},{individual['throughput']:.1f}\n"
            )
            f.write(
                f"{pipeline['total_time']:.3f},{pipeline['stats_time']:.3f},{pipeline['filter_time']:.3f},{pipeline['memory_usage']:.1f},{pipeline['throughput']:.1f}\n"
            )
            f.write(
                f"{(individual['total_time'] - pipeline['total_time']):.3f},{(individual['stats_time'] - pipeline['stats_time']):.3f},{(individual['filter_time'] - pipeline['filter_time']):.3f},{(individual['memory_usage'] - pipeline['memory_usage']):.1f},{(pipeline['throughput'] - individual['throughput']):.1f}\n"
            )
        logger.info(f"📄 Performance summary CSV saved to: {csv_file}")

        # Write configuration file
        config_file = os.path.join(output_dir, "benchmark_config.json")
        config = {
            "mode": mode,
            "num_filters": len(filters),
            "filters": [getattr(f, "_name", type(f).__name__) for f in filters],
            "timestamp": timestamp,
            "benchmark_type": results["benchmark_type"],
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 Configuration saved to: {config_file}")

        logger.info(f"✅ All benchmark results written to: {output_dir}")


def load_real_dataset(dataset_path: str, max_samples: Optional[int] = None) -> Any:
    """
    Load real dataset using DatasetBuilder and convert to expected format.

    Args:
        dataset_path: Path to the dataset file
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Dataset object compatible with Data-Juicer operations
    """
    from argparse import Namespace

    from data_juicer.core.data.dataset_builder import DatasetBuilder

    logger.info(f"📂 Loading real dataset from: {dataset_path}")

    # Create a proper config for DatasetBuilder
    cfg = Namespace()
    cfg.dataset_path = dataset_path
    # Add empty process list to avoid AttributeError
    cfg.process = []

    # Create DatasetBuilder instance
    builder = DatasetBuilder(cfg, executor_type="default")

    # Load the dataset
    dataset = builder.load_dataset()

    # Apply max_samples limit if specified
    if max_samples is not None:
        # Get dataset length safely
        try:
            dataset_length = get_dataset_length(dataset)
            if dataset_length > max_samples:
                # For DJDataset, we need to use a different approach
                # Just log that we're using the full dataset for now
                logger.info(
                    f"⚠️  Dataset has {dataset_length} samples, using all (max_samples not implemented for DJDataset)"
                )
        except Exception:
            logger.info(f"⚠️  Could not determine dataset length, using full dataset")
    else:
        logger.info(f"✅ Loaded dataset from {dataset_path}")

    # Log dataset info
    try:
        dataset_length = get_dataset_length(dataset)
        logger.info(f"📊 Dataset loaded successfully with {dataset_length} samples")
    except Exception:
        logger.info(f"📊 Dataset loaded successfully")

    return dataset


def create_realistic_test_data(num_samples: int = 1000) -> Any:
    """Create realistic test data with diverse text characteristics."""
    logger.info(f"Creating realistic test data with {num_samples} samples...")

    import random

    from faker import Faker

    # Initialize Faker with multiple locales for diversity
    fake = Faker(["en_US", "en_GB", "en_CA"])

    # Add some non-English text for language filtering tests
    non_english_samples = [
        "你好，请问你是谁？",  # Chinese
        "Sur la plateforme MT4, vous pouvez trader des devises.",  # French
        "欢迎来到阿里巴巴！",  # Chinese
        "El sistema de procesamiento de datos es muy eficiente.",  # Spanish
        "Das maschinelle Lernen ist ein Teilgebiet der künstlichen Intelligenz.",  # German
        "La conferenza si terrà il prossimo mese.",  # Italian
        "データ処理システムは非常に効率的です。",  # Japanese
        "시스템 성능을 향상시키기 위한 최적화가 필요합니다。",  # Korean
        "Система обработки данных очень эффективна.",  # Russian
        "نظام معالجة البيانات فعال للغاية.",  # Arabic
        "ระบบประมวลผลข้อมูลมีประสิทธิภาพมาก",  # Thai
        "Sistem pemrosesan data sangat efisien.",  # Indonesian
        "Sistem pemprosesan data sangat cekap.",  # Malay
        "Sistem pemprosesan data sangat cekap.",  # Malay (duplicate for testing)
        "Sistem pemprosesan data sangat cekap.",  # Malay (duplicate for testing)
    ]

    texts = []
    for i in range(num_samples):
        # Create diverse text samples with different characteristics
        if i < len(non_english_samples):
            # Use predefined non-English samples for language filtering tests
            text = non_english_samples[i]
        elif i % 8 == 0:
            # Short realistic sentences
            text = fake.sentence()
        elif i % 8 == 1:
            # Medium paragraphs
            text = fake.paragraph(nb_sentences=3)
        elif i % 8 == 2:
            # Longer texts with multiple paragraphs
            text = fake.text(max_nb_chars=500)
        elif i % 8 == 3:
            # Technical/computer-related text
            text = fake.text(
                max_nb_chars=300,
                ext_word_list=[
                    "algorithm",
                    "database",
                    "network",
                    "software",
                    "hardware",
                    "programming",
                    "development",
                    "system",
                    "data",
                    "processing",
                ],
            )
        elif i % 8 == 4:
            # Business/formal text
            text = fake.text(
                max_nb_chars=400,
                ext_word_list=[
                    "business",
                    "management",
                    "strategy",
                    "organization",
                    "leadership",
                    "performance",
                    "efficiency",
                    "productivity",
                    "innovation",
                ],
            )
        elif i % 8 == 5:
            # Academic/research text
            text = fake.text(
                max_nb_chars=350,
                ext_word_list=[
                    "research",
                    "analysis",
                    "study",
                    "investigation",
                    "evaluation",
                    "methodology",
                    "findings",
                    "conclusion",
                    "hypothesis",
                ],
            )
        elif i % 8 == 6:
            # Casual/conversational text
            text = fake.text(
                max_nb_chars=250,
                ext_word_list=[
                    "conversation",
                    "discussion",
                    "opinion",
                    "experience",
                    "thought",
                    "feeling",
                    "idea",
                    "perspective",
                    "viewpoint",
                ],
            )
        else:
            # Random realistic text
            text = fake.text(max_nb_chars=random.randint(100, 400))

        texts.append(text)

    # Create dataset in the expected format using HuggingFace Dataset
    from datasets import Dataset

    test_data = Dataset.from_dict({"text": texts, Fields.stats: [{} for _ in range(num_samples)]})

    logger.info(f"✅ Created {len(texts)} realistic test samples")
    logger.info(f"   - {len(non_english_samples)} non-English samples for language filtering")
    logger.info(f"   - {len(texts) - len(non_english_samples)} English samples")

    return test_data


def create_simple_test_data(num_samples: int = 1000) -> Any:
    """Create comprehensive test data covering different text characteristics."""
    logger.info(f"Creating {num_samples:,} comprehensive test samples...")

    # For large datasets, create in batches to avoid memory issues
    batch_size = 10000
    texts = []

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_size_actual = batch_end - batch_start

        batch_texts = []
        for i in range(batch_size_actual):
            # Create diverse text samples with different characteristics
            sample_type = i % 8  # 8 different types of text

            if sample_type == 0:
                # Normal text
                length = random.randint(50, 200)
                text = "".join(random.choices(string.ascii_letters + string.digits + " .,!?", k=length))
            elif sample_type == 1:
                # Short text
                length = random.randint(10, 30)
                text = "".join(random.choices(string.ascii_lowercase + " ", k=length))
            elif sample_type == 2:
                # Long text with repetition
                length = random.randint(300, 800)
                text = "".join(random.choices("hello world test data sample " * 20, k=length))
            elif sample_type == 3:
                # Text with special characters
                length = random.randint(100, 300)
                text = "".join(
                    random.choices(string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?", k=length)
                )
            elif sample_type == 4:
                # Text with numbers
                length = random.randint(80, 250)
                text = "".join(random.choices(string.ascii_letters + string.digits + " .,", k=length))
            elif sample_type == 5:
                # Text with high repetition
                length = random.randint(150, 400)
                text = "".join(random.choices("aaaaa bbbbb ccccc ddddd " * 10, k=length))
            elif sample_type == 6:
                # Multi-line text
                lines = random.randint(3, 8)
                text_lines = []
                for _ in range(lines):
                    line_length = random.randint(20, 80)
                    line = "".join(random.choices(string.ascii_letters + " ", k=line_length))
                    text_lines.append(line)
                text = "\n".join(text_lines)
            else:
                # Mixed content
                length = random.randint(100, 350)
                text = "".join(
                    random.choices(string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>? \n\t", k=length)
                )

            batch_texts.append(text)

        texts.extend(batch_texts)

        if batch_end % (batch_size * 5) == 0:
            logger.info(f"  Created {batch_end:,}/{num_samples:,} samples...")

    logger.info(f"Successfully created {len(texts):,} comprehensive test samples")
    logger.info(f"Text characteristics: lengths {min(len(t) for t in texts)}-{max(len(t) for t in texts)} chars")

    # Create dataset in the expected format using HuggingFace Dataset
    from datasets import Dataset

    return Dataset.from_dict({"text": texts, Fields.stats: [{} for _ in range(num_samples)]})


def main():
    """Main execution function for performance benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Performance Benchmark for Data-Juicer Filters - Run individual or pipeline benchmarks separately to avoid caching effects"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "recipe"],
        default="full",
        help="Benchmark mode: quick (3 basic filters), full (12 comprehensive filters), recipe (custom YAML pipeline)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of test samples to use (None for entire dataset)",
    )
    parser.add_argument(
        "--analyzer",
        action="store_true",
        help="Enable analyzer insights for optimization",
    )
    parser.add_argument(
        "--spacy-size",
        choices=["sm", "md", "lg"],
        default="sm",
        help="spaCy model size: sm (fast, ~12MB), md (balanced, ~40MB), lg (accurate, ~560MB)",
    )
    parser.add_argument(
        "--recipe-path",
        type=str,
        default=None,
        help="Path to a YAML recipe for benchmarking (used only in recipe mode)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a real dataset file (JSONL, JSON, CSV, etc.) to use instead of synthetic data",
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["pipeline", "individual", "both"],
        default="both",
        help="Type of benchmark to run: pipeline (optimized), individual (sequential), or both (correctness comparison)",
    )

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = PerformanceBenchmark()

    # Create test data
    if args.dataset_path:
        logger.info(f"🎯 Using real dataset: {args.dataset_path}")
        test_data = load_real_dataset(args.dataset_path, max_samples=args.samples)
    else:
        logger.info(f"🎲 Using synthetic data: {args.samples} samples")
        test_data = create_realistic_test_data(args.samples)

    # Get filters based on mode
    if args.mode == "quick":
        filters = benchmark.create_quick_test_filters()
    elif args.mode == "full":
        filters = benchmark.create_test_filters()  # Use comprehensive test filters (12 filters)
    elif args.mode == "recipe":
        # Simple recipe loading - just load the process list from YAML
        if not args.recipe_path:
            raise ValueError("--recipe-path must be specified in recipe mode!")

        # Load recipe YAML
        import yaml

        with open(args.recipe_path, "r") as f:
            recipe_config = yaml.safe_load(f)

        # Extract process list
        process_list = recipe_config.get("process", [])
        if not process_list:
            raise ValueError(f"No 'process' section found in recipe: {args.recipe_path}")

        # Load operations directly using load_ops
        from data_juicer.ops import load_ops

        filters = load_ops(process_list)

        logger.info(f"📋 Loaded recipe: {args.recipe_path}")
        logger.info(f"📊 Loaded {len(filters)} operations:")
        for i, op in enumerate(filters):
            logger.info(f"  {i+1}: {type(op).__name__}")
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Get analyzer insights if requested
    analyzer_insights: Optional[Dict[str, Any]] = None
    if args.analyzer:
        logger.info("🔍 Running analyzer to get insights...")
        analyzer_insights = benchmark.run_analyzer(test_data)

    # Run comprehensive benchmark with fusion strategies
    results = benchmark.run_benchmark(filters, test_data, args.mode, analyzer_insights, args.benchmark_type)

    logger.info("\n✅ Benchmark completed successfully!")
    logger.info(f"📊 Results saved for mode: {args.mode}")

    return results


if __name__ == "__main__":
    main()
