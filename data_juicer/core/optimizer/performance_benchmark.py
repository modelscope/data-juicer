#!/usr/bin/env python3
"""
Performance benchmark for Data-Juicer filter fusion and optimization.

This benchmark compares individual vs fused filter performance and demonstrates
the new PipelineOptimizer architecture. The optimizer architecture with analyzer
insights is used by default in all modes.

USAGE EXAMPLES:
    # Full comprehensive benchmark (default) - uses optimizer by default
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

MODES:
    full     - Comprehensive benchmark with optimizer architecture (default)
    quick    - Basic performance demo with optimizer architecture
    recipe   - Benchmark a real YAML pipeline (requires --recipe-path)
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
from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy
from data_juicer.core.optimizer.fused_op import FusedFilter, FusedMapper
from data_juicer.core.optimizer.mapper_fusion_strategy import MapperFusionStrategy
from data_juicer.core.optimizer.optimizer import PipelineOptimizer
from data_juicer.core.pipeline_ast import PipelineAST
from data_juicer.ops import load_ops
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

    def create_realistic_test_data(self, num_samples: int = 1000) -> Dict[str, Any]:
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

        # Create dataset in the expected format
        test_data = {"text": texts, Fields.stats: [{} for _ in range(num_samples)]}

        logger.info(f"✅ Created {len(texts)} realistic test samples")
        logger.info(f"   - {len(non_english_samples)} non-English samples for language filtering")
        logger.info(f"   - {len(texts) - len(non_english_samples)} English samples")

        return test_data

    def create_test_filters(self) -> List[Filter]:
        """Create a comprehensive set of test filters covering different categories."""
        logger.info("Creating comprehensive test filters...")

        filters = [
            # Basic text filters (simple, fast) - Made extremely lenient
            WordsNumFilter(min_num=1, max_num=100000),  # Allow 1-100000 words
            TextLengthFilter(min_len=1, max_len=100000),  # Allow 1-100000 chars
            CharacterRepetitionFilter(repetition_ratio=0.999),  # Allow up to 99.9% repetition
            WordRepetitionFilter(min_ratio=0.0, max_ratio=0.99),  # Allow up to 99% word repetition
            SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.95),  # Allow up to 95% special chars
            AlphanumericFilter(min_ratio=0.001),  # Allow as low as 0.1% alphanumeric
            AverageLineLengthFilter(min_len=1, max_len=10000),  # Allow 1-10000 chars per line
            MaximumLineLengthFilter(min_len=1, max_len=20000),  # Allow 1-20000 chars per line
            # Content quality filters (moderate complexity) - Made extremely lenient
            PerplexityFilter(
                lang="en", model_key="gpt2", min_score=0.0, max_score=1e10
            ),  # Allow extremely high perplexity (was 1e6)
            StopWordsFilter(lang="en", min_ratio=0.0, max_ratio=0.99),  # Allow up to 99% stop words
            FlaggedWordFilter(lang="en", min_ratio=0.0, max_ratio=0.99),  # Allow up to 99% flagged words
            LanguageIDScoreFilter(lang="en", min_score=0.0),  # Allow any language score
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

    def run_individual_filters_benchmark(self, filters: List[Filter], test_data: Dict[str, Any]) -> PerformanceMetrics:
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

            # Compute stats for this filter
            stats_start = time.time()
            if hasattr(filter_op, "compute_stats_batched"):
                samples_with_stats = filter_op.compute_stats_batched(samples_with_stats)
            stats_time = time.time() - stats_start
            total_stats_time += stats_time

            # Immediately filter with this filter
            filter_start = time.time()
            if hasattr(filter_op, "process_batched"):
                result = list(filter_op.process_batched(samples_with_stats))

                # DEBUG: Log filter result details
                if result:
                    result_type = type(result[0]).__name__
                    result_count = len(result)
                    if isinstance(result[0], bool):
                        passed_count = sum(result)
                        logger.info(
                            f"🔍 DEBUG INDIVIDUAL: Filter {i+1} ({filter_name}) returned {result_count} booleans: {passed_count} passed, {result_count - passed_count} failed"
                        )
                        logger.debug(f"🔍 DEBUG INDIVIDUAL: Filter {i+1} mask (first 10): {result[:10]}")
                    else:
                        logger.info(
                            f"🔍 DEBUG INDIVIDUAL: Filter {i+1} ({filter_name}) returned {result_count} {result_type} items (likely mapper)"
                        )
                else:
                    logger.debug(f"🔍 DEBUG INDIVIDUAL: Filter {i+1} ({filter_name}) returned empty result")

            filter_time = time.time() - filter_start
            total_filter_time += filter_time

            logger.debug(f"      Filter {i+1} - Stats: {stats_time:.3f}s, Filter: {filter_time:.3f}s")

        processing_time = time.time() - processing_start
        logger.info(f"  Step 2 - Complete processing: {processing_time:.3f}s")
        logger.info(f"    Total stats time: {total_stats_time:.3f}s")
        logger.info(f"    Total filter time: {total_filter_time:.3f}s")

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = get_dataset_length(test_data) / total_time

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
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Benchmark the complete pipeline optimizer workflow."""
        logger.info("Running pipeline optimizer benchmark (complete workflow)...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Step 1: Build pipeline configuration from filters
        pipeline_config = self._build_pipeline_config_from_filters(filters)

        # Step 2: Create Pipeline AST
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        # Step 3: Create PipelineOptimizer with fusion strategies

        strategies = [FilterFusionStrategy(analyzer_insights=analyzer_insights)]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)

        # Step 4: Get optimization summary
        optimization_summary = optimizer.get_optimization_summary()
        logger.info("  Pipeline Optimizer Configuration:")
        logger.info(f"    Strategies: {optimization_summary['strategies']}")
        logger.info(f"    Analyzer insights: {optimization_summary['analyzer_insights_available']}")

        # Step 5: Apply optimizations
        logger.info("  Applying pipeline optimizations...")
        optimized_ast = optimizer.optimize(ast)

        # Step 6: Convert optimized AST back to operations
        optimized_ops = self._convert_ast_to_operations(optimized_ast)
        logger.info(f"  Original operations: {len(filters)}")
        logger.info(f"  Optimized operations: {len(optimized_ops)}")

        # Step 7: Process with optimized operations
        logger.info("  Processing with optimized pipeline...")
        self._process_with_optimized_ops(optimized_ops, test_data)

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = get_dataset_length(test_data) / total_time

        logger.info("  📊 PIPELINE OPTIMIZER BREAKDOWN:")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")
        logger.info(f"    Memory usage: {memory_usage:.1f} MB")
        logger.info(f"    Optimization ratio: {len(optimized_ops)/len(filters):.2f}x")
        logger.info(f"    Operations reduced: {len(filters) - len(optimized_ops)}")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.8,  # Estimate: most time is processing
            filter_time=total_time * 0.2,  # Estimate: some time is optimization
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def _build_pipeline_config_from_filters(self, filters: List[Filter]) -> Dict[str, Any]:
        process_config = []
        for i, filter_op in enumerate(filters):
            op_name = getattr(filter_op, "_name", f"filter_{i}")
            op_config = getattr(filter_op, "config", None)
            if op_config:
                process_config.append({op_name: op_config})
            else:
                # Create basic config from filter attributes
                config_dict = {}
                for attr in dir(filter_op):
                    if not attr.startswith("_") and not callable(getattr(filter_op, attr)):
                        value = getattr(filter_op, attr)
                        if isinstance(value, (int, float, str, bool, list)):
                            config_dict[attr] = value
                process_config.append({op_name: config_dict})
        return {"process": process_config}

    def _convert_ast_to_operations(self, ast) -> List:
        """Convert optimized AST back to operations list."""
        # This is a simplified conversion - in practice, you'd need more sophisticated logic
        operations = []

        def traverse_node(node):
            if hasattr(node, "children") and node.children:
                for child in node.children:
                    # Skip root node
                    if child.name == "root":
                        traverse_node(child)
                        continue

                    # Extract operation name and config from the node
                    op_name = child.name
                    op_config = child.config if child.config else {}

                    # Debug: Log what we're processing
                    logger.debug(f"Processing node: {op_name} with config: {op_config}")

                    # Handle different operation types
                    if op_name == "fused_mapper":
                        # Extract fused mapper configuration - handle double nesting
                        if "fused_mapper" in op_config and isinstance(op_config["fused_mapper"], dict):
                            mapper_config = op_config["fused_mapper"]
                            # Check if there's another level of nesting
                            if "fused_mapper" in mapper_config:
                                mapper_config = mapper_config["fused_mapper"]
                            clean_config = {
                                "name": mapper_config.get("name", "fused_mapper"),
                                "fused_mappers": mapper_config.get("fused_mappers", []),
                            }
                            operations.append({op_name: clean_config})
                        else:
                            # Fallback if structure is different
                            operations.append({op_name: op_config})

                    elif op_name == "fused_filter":
                        # Extract fused filter configuration
                        fused_op_list = None
                        # Handle both possible config structures
                        if "fused_op_list" in op_config:
                            fused_op_list = op_config["fused_op_list"]
                            logger.debug(f"Found fused_op_list in op_config: {len(fused_op_list)} items")
                        elif "general_fused_op" in op_config and "fused_op_list" in op_config["general_fused_op"]:
                            fused_op_list = op_config["general_fused_op"]["fused_op_list"]
                            logger.debug(f"Found fused_op_list in general_fused_op: {len(fused_op_list)} items")
                        else:
                            logger.debug(
                                f"No fused_op_list found in op_config. Available keys: {list(op_config.keys())}"
                            )
                        if fused_op_list is not None:
                            clean_config = {"fused_op_list": fused_op_list}
                            operations.append({op_name: clean_config})
                            logger.debug(f"Added fused_filter with {len(fused_op_list)} operations")
                        else:
                            # Fallback if structure is different
                            operations.append({op_name: op_config})
                            logger.debug("Added fused_filter with fallback config (no fused_op_list)")

                    else:
                        # Generic handling for all other operation types
                        # This includes filters, mappers, deduplicators, etc.
                        operations.append({op_name: op_config})
                        logger.debug(f"Added operation: {op_name}")

                    # Continue traversing
                    traverse_node(child)

        if ast.root:
            traverse_node(ast.root)

        return operations

    def _process_with_optimized_ops(self, optimized_ops: List, test_data: Dict[str, Any]):
        """Process test data with optimized operations."""
        logger.debug(f"Processing with {len(optimized_ops)} optimized operations")

        # Load and execute the optimized operations
        from data_juicer.ops import load_ops

        # DEBUG: Track sample counts and operation results
        original_sample_count = get_dataset_length(test_data)
        logger.debug(f"🔍 DEBUG PIPELINE: Starting with {original_sample_count} samples")

        data = test_data
        for op_idx, op_config in enumerate(optimized_ops):
            # Debug: Log the operation configuration
            logger.debug(f"Loading operation config: {op_config}")

            # Special handling for fused_filter - we need to create the actual filter objects
            op_name = list(op_config.keys())[0]
            logger.debug(f"🔍 DEBUG PIPELINE: Processing op {op_idx+1}/{len(optimized_ops)}: {op_name}")

            # DEBUG: Log current state before operation
            current_sample_count = get_dataset_length(data)
            logger.debug(f"🔍 DEBUG PIPELINE: Before op {op_idx+1} ({op_name}): {current_sample_count} samples")

            if op_name == "fused_filter":
                # Extract the fused_op_list and create individual filter objects
                fused_op_list = op_config[op_name].get("fused_op_list", [])
                individual_filters = []

                logger.debug(f"🔍 DEBUG PIPELINE: Fused filter contains {len(fused_op_list)} individual filters:")
                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]
                    logger.info(f"    - {filter_name}: {filter_args}")

                    # Load the individual filter
                    loaded_filters = load_ops([{filter_name: filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])
                        logger.info(f"    ✅ Successfully loaded {filter_name}")
                    else:
                        logger.warning(f"    ❌ Failed to load {filter_name}")

                # Create the fused filter with the actual filter objects
                if individual_filters:
                    fused_filter = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force parallel execution to match individual execution behavior
                    # (each filter sees original data, not filtered output from previous filters)
                    fused_filter.execution_strategy = "parallel"
                    logger.debug(f"🔍 DEBUG PIPELINE: Created fused filter with {len(individual_filters)} filters")
                    logger.info(
                        f"🔍 DEBUG PIPELINE: Fused filter execution strategy: {fused_filter.execution_strategy}"
                    )
                    logger.debug(f"🔍 DEBUG PIPELINE: Filter order: {[f._name for f in fused_filter.fused_filters]}")

                    # Process the fused filter
                    if hasattr(fused_filter, "compute_stats_batched"):
                        data = fused_filter.compute_stats_batched(data)
                    if hasattr(fused_filter, "process_batched"):
                        result = list(fused_filter.process_batched(data))

                        # DEBUG: Log fused filter result details
                        if result:
                            result_type = type(result[0]).__name__
                            result_count = len(result)
                            if isinstance(result[0], bool):
                                passed_count = sum(result)
                                logger.info(
                                    f"🔍 DEBUG PIPELINE: Fused filter returned {result_count} booleans: {passed_count} passed, {result_count - passed_count} failed"
                                )
                                logger.debug(f"🔍 DEBUG PIPELINE: Fused filter mask (first 10): {result[:10]}")
                            else:
                                logger.info(
                                    f"🔍 DEBUG PIPELINE: Fused filter returned {result_count} {result_type} items (likely mapper)"
                                )
                        else:
                            logger.debug(f"🔍 DEBUG PIPELINE: Fused filter returned empty result")
                else:
                    logger.warning(f"🔍 DEBUG PIPELINE: Failed to create fused filter")
                    continue  # Skip if we can't create the fused filter
            else:
                # Load the operation from config for non-fused operations
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    op = loaded_ops[0]
                    logger.debug(f"🔍 DEBUG PIPELINE: Loaded op: {type(op).__name__}")

                    # Execute the operation
                    if hasattr(op, "compute_stats_batched"):
                        data = op.compute_stats_batched(data)
                    if hasattr(op, "process_batched"):
                        result = list(op.process_batched(data))

                        # DEBUG: Log operation result details
                        if result:
                            result_type = type(result[0]).__name__
                            result_count = len(result)
                            if isinstance(result[0], bool):
                                passed_count = sum(result)
                                logger.info(
                                    f"🔍 DEBUG PIPELINE: Op {op_idx+1} ({op_name}) returned {result_count} booleans: {passed_count} passed, {result_count - passed_count} failed"
                                )
                                logger.debug(f"🔍 DEBUG PIPELINE: Op {op_idx+1} mask (first 10): {result[:10]}")
                            else:
                                logger.info(
                                    f"🔍 DEBUG PIPELINE: Op {op_idx+1} ({op_name}) returned {result_count} {result_type} items (likely mapper)"
                                )
                        else:
                            logger.debug(f"🔍 DEBUG PIPELINE: Op {op_idx+1} ({op_name}) returned empty result")
                else:
                    logger.warning(f"🔍 DEBUG PIPELINE: Failed to load op from config")
                    continue  # Skip if we can't load the operation

            # DEBUG: Log current state after operation
            current_sample_count = get_dataset_length(data)
            logger.debug(f"🔍 DEBUG PIPELINE: After op {op_idx+1} ({op_name}): {current_sample_count} samples")

    def _extract_original_filters_from_fused(self, filters):
        """Extract original individual filters from fused filters for validation."""
        original_filters = []
        for filter_op in filters:
            if hasattr(filter_op, "_name") and filter_op._name == "fused_filter":
                # This is a fused filter, extract the individual filters
                if hasattr(filter_op, "fused_filters"):
                    original_filters.extend(filter_op.fused_filters)
            else:
                # This is an individual filter, keep it as is
                original_filters.append(filter_op)

        # If no original filters were found, return the original list
        # (this handles the case where filters are already individual)
        if not original_filters:
            return filters

        return original_filters

    def get_final_mask_from_filters(self, filters: List[Filter], test_data: Dict[str, Any]) -> list:
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

    def get_final_mask_from_optimized_ops(self, optimized_ops: List, test_data: Dict[str, Any]) -> list:
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
        test_data: Dict[str, Any],
        mode: str = "quick",
        analyzer_insights: dict = None,
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark comparing individual vs pipeline optimizer execution."""
        logger.info("🚀 Starting Performance Benchmark")
        logger.info("=" * 60)

        # Get analyzer insights if not provided
        if analyzer_insights is None:
            logger.info("🔍 Getting analyzer insights...")
            analyzer_insights = self.get_analyzer_insights(test_data)

        # Run individual filters benchmark
        logger.info("\n📊 INDIVIDUAL EXECUTION BENCHMARK")
        logger.info("-" * 40)
        individual_results = self.run_individual_filters_benchmark(filters, test_data)

        # Run pipeline optimizer benchmark
        logger.info("\n🔧 PIPELINE OPTIMIZER BENCHMARK")
        logger.info("-" * 40)
        pipeline_results = self.run_pipeline_optimizer_benchmark(filters, test_data, analyzer_insights)

        # Calculate performance metrics
        individual_time = individual_results.total_time
        pipeline_time = pipeline_results.total_time

        if individual_time > 0:
            pipeline_speedup = individual_time / pipeline_time
            logger.info(f"Individual execution: {individual_time:.3f}s ({individual_results.throughput:.1f} samples/s)")
            logger.info(f"Pipeline optimizer:   {pipeline_time:.3f}s ({pipeline_results.throughput:.1f} samples/s)")
            logger.info(f"Pipeline speedup:     {pipeline_speedup:.2f}x")

            # Determine best strategy
            best_time = min(individual_time, pipeline_time)
            if best_time == individual_time:
                best_strategy = "Individual"
            else:
                best_strategy = "Pipeline Optimizer"

            logger.info(f"🏆 Best strategy:      {best_strategy}")

        # --- Simple Validation: Save and Compare Results ---
        logger.info("\n🔎 VALIDATING PIPELINE RESULTS AGAINST INDIVIDUAL EXECUTION")

        # Create validation directory in outputs/
        validation_dir = f"./outputs/benchmark_validation_{mode}_{int(time.time())}"
        os.makedirs(validation_dir, exist_ok=True)
        logger.info(f"📁 Validation results will be saved to: {validation_dir}")

        # Get the optimized operations from the pipeline benchmark
        pipeline_config = self._build_pipeline_config_from_filters(filters)
        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        strategies = [FilterFusionStrategy()]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)
        optimized_ast = optimizer.optimize(ast)
        optimized_ops = self._convert_ast_to_operations(optimized_ast)

        # Debug: Log what we're comparing
        logger.info(
            f"🔍 VALIDATION DEBUG: Comparing {len(filters)} filters vs {len(optimized_ops)} optimized operations"
        )
        for i, f in enumerate(filters):
            logger.info(f"  Filter {i+1}: {type(f).__name__}")
        for i, op in enumerate(optimized_ops):
            op_name = list(op.keys())[0]
            logger.info(f"  Optimized Op {i+1}: {op_name}")

        # For validation, we need to compare the original individual filters to the optimized pipeline
        # In recipe mode, 'filters' might contain fused filters, so we need to extract the original filters
        original_filters = self._extract_original_filters_from_fused(filters)
        logger.debug(f"🔍 VALIDATION DEBUG: Extracted {len(original_filters)} original filters for validation")

        # Run both individual and pipeline execution and save results
        individual_results_data = self._run_and_save_individual_execution(original_filters, test_data, validation_dir)
        pipeline_results_data = self._run_and_save_pipeline_execution(optimized_ops, test_data, validation_dir)

        # Compare results
        validation_results = self._compare_execution_results(
            individual_results_data, pipeline_results_data, validation_dir
        )

        # Compile results
        results = {
            "mode": mode,
            "num_samples": get_dataset_length(test_data),
            "num_filters": len(original_filters),  # Use original filter count for reporting
            "individual": {
                "total_time": individual_results.total_time,
                "stats_time": individual_results.stats_time,
                "filter_time": individual_results.filter_time,
                "memory_usage": individual_results.memory_usage,
                "throughput": individual_results.throughput,
            },
            "pipeline": {
                "total_time": pipeline_results.total_time,
                "stats_time": pipeline_results.stats_time,
                "filter_time": pipeline_results.filter_time,
                "memory_usage": pipeline_results.memory_usage,
                "throughput": pipeline_results.throughput,
            },
            "speedup": pipeline_speedup if individual_time > 0 else 0,
            "best_strategy": best_strategy if individual_time > 0 else "Unknown",
            "validation": validation_results,
            "analyzer_insights": analyzer_insights,
            "validation_dir": validation_dir,
        }

        return results

    def _run_and_save_individual_execution(
        self, filters: List[Filter], test_data: Dict[str, Any], validation_dir: str
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
            if data.get("text") and len(data["text"]) > 0:
                logger.debug(f"🔍 DEBUG: First text sample before filter {i+1}: {data['text'][0][:50]}...")

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
                        logger.debug(f"🔍 DEBUG: After filter {i+1}: 0 samples remaining")
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
        self, optimized_ops: List, test_data: Dict[str, Any], validation_dir: str
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
                    logger.info(
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
            alphanumeric_filter.AlphanumericFilter(min_ratio=0.5),
            text_length_filter.TextLengthFilter(min_len=10, max_len=1000),
            words_num_filter.WordsNumFilter(min_num=5, max_num=200),
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
            alphanumeric_filter.AlphanumericFilter(min_ratio=0.01),  # Allow as low as 1% alphanumeric (was 0.1)
            text_length_filter.TextLengthFilter(min_len=1, max_len=50000),  # Allow 1-50000 chars (was 1-10000)
            words_num_filter.WordsNumFilter(min_num=1, max_num=50000),  # Allow 1-50000 words (was 1-10000)
            character_repetition_filter.CharacterRepetitionFilter(
                max_ratio=0.99
            ),  # Allow up to 99% repetition (was 0.95)
            word_repetition_filter.WordRepetitionFilter(max_ratio=0.95),  # Allow up to 95% word repetition (was 0.8)
            special_characters_filter.SpecialCharactersFilter(
                min_ratio=0.0, max_ratio=0.8
            ),  # Allow up to 80% special chars (was 0.0-0.5)
            stopwords_filter.StopWordsFilter(min_ratio=0.0),  # Allow any stop word ratio (was 0.0)
        ]

    def run_analyzer(self, test_data: Dict[str, Any]) -> dict:
        """Run analyzer to get insights for optimization."""
        try:
            from data_juicer.analysis import Analyzer

            analyzer = Analyzer()
            insights = analyzer.analyze(test_data)
            logger.info("✅ Analyzer insights generated successfully")
            return insights
        except Exception as e:
            logger.warning(f"⚠️  Analyzer failed: {e}")
            return None

    def run_ast_benchmark(self, ast, test_data: Dict[str, Any], mode: str = "recipe") -> Dict[str, Any]:
        """Run benchmark using AST-based execution for recipe mode."""
        logger.info("🚀 Starting AST-based Performance Benchmark")
        logger.info("=" * 60)

        # Get analyzer insights
        logger.info("🔍 Getting analyzer insights...")
        analyzer_insights = self.get_analyzer_insights(test_data)

        # Run individual execution benchmark (convert AST to individual operations)
        logger.info("\n📊 INDIVIDUAL EXECUTION BENCHMARK")
        logger.info("-" * 40)

        # Convert AST to individual operations for baseline comparison
        individual_ops = self._convert_ast_to_individual_ops(ast)
        individual_results = self.run_individual_filters_benchmark(individual_ops, test_data)

        # Run AST-based pipeline benchmark
        logger.info("\n🔧 AST PIPELINE BENCHMARK")
        logger.info("-" * 40)
        pipeline_results = self.run_ast_pipeline_benchmark(ast, test_data, analyzer_insights)

        # Calculate performance metrics
        individual_time = individual_results.total_time
        pipeline_time = pipeline_results.total_time

        if individual_time > 0:
            pipeline_speedup = individual_time / pipeline_time
            logger.info(f"Individual execution: {individual_time:.3f}s ({individual_results.throughput:.1f} samples/s)")
            logger.info(f"AST pipeline:        {pipeline_time:.3f}s ({pipeline_results.throughput:.1f} samples/s)")
            logger.info(f"Pipeline speedup:    {pipeline_speedup:.2f}x")

            # Determine best strategy
            best_time = min(individual_time, pipeline_time)
            if best_time == individual_time:
                best_strategy = "Individual"
            else:
                best_strategy = "AST Pipeline"

            logger.info(f"🏆 Best strategy:     {best_strategy}")

        # Validation: Compare AST execution vs individual execution
        logger.info("\n🔎 VALIDATING AST PIPELINE RESULTS AGAINST INDIVIDUAL EXECUTION")

        # Get final masks for comparison
        individual_mask = self.get_final_mask_from_filters(individual_ops, test_data)
        ast_mask = self.get_final_mask_from_ast(ast, test_data)

        if individual_mask == ast_mask:
            logger.info("✅ AST pipeline results match individual execution!")
            mismatches = []
        else:
            mismatches = [i for i, (a, b) in enumerate(zip(individual_mask, ast_mask)) if a != b]
            logger.warning(
                f"❌ AST pipeline results do NOT match individual execution! {len(mismatches)} mismatches out of {len(individual_mask)} samples."
            )

        # Compile results
        results = {
            "mode": mode,
            "num_samples": get_dataset_length(test_data),
            "num_filters": len(individual_ops),
            "individual": {
                "total_time": individual_results.total_time,
                "stats_time": individual_results.stats_time,
                "filter_time": individual_results.filter_time,
                "memory_usage": individual_results.memory_usage,
                "throughput": individual_results.throughput,
            },
            "pipeline": {
                "total_time": pipeline_results.total_time,
                "stats_time": pipeline_results.stats_time,
                "filter_time": pipeline_results.filter_time,
                "memory_usage": pipeline_results.memory_usage,
                "throughput": pipeline_results.throughput,
            },
            "speedup": pipeline_speedup if individual_time > 0 else 0,
            "best_strategy": best_strategy if individual_time > 0 else "Unknown",
            "validation": {
                "matches": len(mismatches) == 0,
                "num_mismatches": len(mismatches),
                "mismatch_indices": mismatches[:10],
            },
            "analyzer_insights": analyzer_insights,
        }

        return results

    def run_ast_pipeline_benchmark(
        self, ast, test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Benchmark AST-based pipeline execution."""
        logger.info("Running AST pipeline benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Convert AST back to config for execution
        config = self._convert_ast_to_config(ast)

        # Create a temporary config for execution
        from jsonargparse import Namespace

        from data_juicer.core.executor import DefaultExecutor

        # Create minimal config for execution
        exec_config = Namespace()
        exec_config.process = config.get("process", [])
        exec_config.work_dir = "./tmp_benchmark"
        exec_config.export_path = "./tmp_benchmark/result.jsonl"
        exec_config.export_shard_size = 10000
        exec_config.export_in_parallel = False
        exec_config.np = 1
        exec_config.use_cache = False
        exec_config.use_checkpoint = False
        exec_config.open_monitor = False
        exec_config.open_tracer = False
        exec_config.op_fusion = False
        exec_config.adaptive_batch_size = False
        exec_config.keep_stats_in_res_ds = True
        exec_config.keep_hashes_in_res_ds = False

        # Create executor and run
        executor = DefaultExecutor(exec_config)
        _ = executor.run(dataset=test_data, skip_return=False)

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = get_dataset_length(test_data) / total_time

        logger.info("  📊 AST PIPELINE BREAKDOWN:")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")
        logger.info(f"    Memory usage: {memory_usage:.1f} MB")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.8,  # Estimate
            filter_time=total_time * 0.2,  # Estimate
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def _convert_ast_to_config(self, ast) -> Dict[str, Any]:
        """Convert AST back to configuration format."""
        if not ast.root or not ast.root.children:
            return {"process": []}

        process_list = []
        current = ast.root.children[0]  # Skip root node

        while current:
            process_list.append({current.name: current.config})
            if current.children:
                current = current.children[0]
            else:
                break

        return {"process": process_list}

    def get_final_mask_from_ast(self, ast, test_data: Dict[str, Any]) -> list:
        """Compute final boolean mask using AST execution."""
        # Convert AST to config and execute
        config = self._convert_ast_to_config(ast)

        # Create a temporary config for execution
        from jsonargparse import Namespace

        from data_juicer.core.executor import DefaultExecutor

        # Create minimal config for execution
        exec_config = Namespace()
        exec_config.process = config.get("process", [])
        exec_config.work_dir = "./tmp_benchmark"
        exec_config.export_path = "./tmp_benchmark/result.jsonl"
        exec_config.export_shard_size = 10000
        exec_config.export_in_parallel = False
        exec_config.np = 1
        exec_config.use_cache = False
        exec_config.use_checkpoint = False
        exec_config.open_monitor = False
        exec_config.open_tracer = False
        exec_config.op_fusion = False
        exec_config.adaptive_batch_size = False
        exec_config.keep_stats_in_res_ds = True
        exec_config.keep_hashes_in_res_ds = False

        # Create executor and run
        executor = DefaultExecutor(exec_config)
        result_dataset = executor.run(dataset=test_data, skip_return=False)

        # Create mask based on which samples survived
        original_length = get_dataset_length(test_data)
        result_length = get_dataset_length(result_dataset)

        # For now, assume all samples that made it through the pipeline passed
        # This is a simplified approach - in practice, you'd need to track which samples were dropped
        mask = [True] * result_length + [False] * (original_length - result_length)

        return mask

    def _convert_ast_to_individual_ops(self, ast) -> List[Filter]:
        """Convert AST to individual operations for baseline comparison."""
        operations = self._convert_ast_to_operations(ast)
        individual_ops = []

        from data_juicer.ops import load_ops

        for op_config in operations:
            op_name = list(op_config.keys())[0]
            if op_name == "fused_filter":
                # Extract individual filters from fused filter
                fused_op_list = op_config[op_name].get("fused_op_list", [])
                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]
                    loaded_filters = load_ops([{filter_name: filter_args}])
                    if loaded_filters:
                        individual_ops.append(loaded_filters[0])
            else:
                # Load individual operation
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    individual_ops.append(loaded_ops[0])

        return individual_ops

    def classify_operations(self, operations: List) -> Dict[str, List]:
        """Classify operations by type for proper benchmarking."""
        classified = {"mappers": [], "filters": [], "deduplicators": [], "other": []}

        for op in operations:
            op_type = type(op).__name__.lower()
            if "mapper" in op_type:
                classified["mappers"].append(op)
            elif "filter" in op_type:
                classified["filters"].append(op)
            elif "deduplicator" in op_type:
                classified["deduplicators"].append(op)
            else:
                classified["other"].append(op)

        return classified

    def run_mixed_operations_benchmark_with_original_ops(
        self, original_operations: List, optimized_operations: List, test_data: Dict[str, Any], mode: str = "mixed"
    ) -> Dict[str, Any]:
        """Run benchmark for mixed operations using original operations for individual execution and optimized operations for pipeline execution."""
        logger.info("🚀 Starting Mixed Operations Performance Benchmark")
        logger.info("=" * 60)

        # Classify operations by type
        original_classified_ops = self.classify_operations(original_operations)
        optimized_classified_ops = self.classify_operations(optimized_operations)

        logger.info("📊 Original operations breakdown:")
        logger.info(f"  Mappers: {len(original_classified_ops['mappers'])}")
        logger.info(f"  Filters: {len(original_classified_ops['filters'])}")
        logger.info(f"  Deduplicators: {len(original_classified_ops['deduplicators'])}")
        logger.info(f"  Other: {len(original_classified_ops['other'])}")

        logger.info("📊 Optimized operations breakdown:")
        logger.info(f"  Mappers: {len(optimized_classified_ops['mappers'])}")
        logger.info(f"  Filters: {len(optimized_classified_ops['filters'])}")
        logger.info(f"  Deduplicators: {len(optimized_classified_ops['deduplicators'])}")
        logger.info(f"  Other: {len(optimized_classified_ops['other'])}")

        # Get analyzer insights
        logger.info("🔍 Getting analyzer insights...")
        analyzer_insights = self.get_analyzer_insights(test_data)

        # Run individual execution benchmark with original operations
        logger.info("\n📊 INDIVIDUAL EXECUTION BENCHMARK")
        logger.info("-" * 40)
        individual_results = self.run_individual_mixed_ops_benchmark(original_operations, test_data)

        # Run pipeline optimizer benchmark with optimized operations
        logger.info("\n🔧 PIPELINE OPTIMIZER BENCHMARK")
        logger.info("-" * 40)
        pipeline_results = self.run_pipeline_mixed_ops_benchmark(optimized_operations, test_data, analyzer_insights)

        # Calculate performance metrics
        individual_time = individual_results.total_time
        pipeline_time = pipeline_results.total_time

        if individual_time > 0:
            pipeline_speedup = individual_time / pipeline_time
            logger.info(f"Individual execution: {individual_time:.3f}s ({individual_results.throughput:.1f} samples/s)")
            logger.info(f"Pipeline optimizer:   {pipeline_time:.3f}s ({pipeline_results.throughput:.1f} samples/s)")
            logger.info(f"Pipeline speedup:     {pipeline_speedup:.2f}x")

            # Determine best strategy
            best_time = min(individual_time, pipeline_time)
            if best_time == individual_time:
                best_strategy = "Individual"
            else:
                best_strategy = "Pipeline Optimizer"

            logger.info(f"🏆 Best strategy:      {best_strategy}")

        # --- Simple Validation: Save and Compare Results ---
        logger.info("\n🔎 VALIDATING PIPELINE RESULTS AGAINST INDIVIDUAL EXECUTION")

        # Create validation directory in outputs/
        validation_dir = f"./outputs/benchmark_validation_{mode}_{int(time.time())}"
        os.makedirs(validation_dir, exist_ok=True)
        logger.info(f"📁 Validation results will be saved to: {validation_dir}")

        # Get the optimized operations from the pipeline benchmark
        pipeline_config = self._build_pipeline_config_from_operations(optimized_operations)
        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )

        strategies = [FilterFusionStrategy(), MapperFusionStrategy()]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)
        optimized_ast = optimizer.optimize(ast)
        _ = self._convert_ast_to_operations(optimized_ast)

        # Debug: Log what we're comparing
        logger.info(
            f"🔍 VALIDATION DEBUG: Comparing {len(original_operations)} original operations vs {len(optimized_operations)} optimized operations"
        )
        for i, op in enumerate(original_operations):
            logger.info(f"  Original Operation {i+1}: {type(op).__name__}")
        for i, op in enumerate(optimized_operations):
            logger.info(f"  Optimized Operation {i+1}: {type(op).__name__}")

        # Run both individual and pipeline execution and save results
        individual_results_data = self._run_and_save_individual_mixed_execution(
            original_operations, test_data, validation_dir
        )
        pipeline_results_data = self._run_and_save_pipeline_mixed_execution(optimized_ast, test_data, validation_dir)

        # Compare results
        validation_results = self._compare_execution_results(
            individual_results_data, pipeline_results_data, validation_dir
        )

        # Compile results
        results = {
            "mode": mode,
            "num_samples": get_dataset_length(test_data),
            "num_original_operations": len(original_operations),
            "num_optimized_operations": len(optimized_operations),
            "original_operation_breakdown": original_classified_ops,
            "optimized_operation_breakdown": optimized_classified_ops,
            "individual": {
                "total_time": individual_results.total_time,
                "stats_time": individual_results.stats_time,
                "filter_time": individual_results.filter_time,
                "memory_usage": individual_results.memory_usage,
                "throughput": individual_results.throughput,
            },
            "pipeline": {
                "total_time": pipeline_results.total_time,
                "stats_time": pipeline_results.stats_time,
                "filter_time": pipeline_results.filter_time,
                "memory_usage": pipeline_results.memory_usage,
                "throughput": pipeline_results.throughput,
            },
            "speedup": pipeline_speedup if individual_time > 0 else 0,
            "best_strategy": best_strategy if individual_time > 0 else "Unknown",
            "validation": validation_results,
            "analyzer_insights": analyzer_insights,
            "validation_dir": validation_dir,
        }

        return results

    def run_mixed_operations_benchmark(
        self, operations: List, test_data: Dict[str, Any], mode: str = "mixed"
    ) -> Dict[str, Any]:
        """Run benchmark for mixed operations (mappers, filters, deduplicators)."""
        # For backward compatibility, use the same operations for both individual and pipeline
        return self.run_mixed_operations_benchmark_with_original_ops(operations, operations, test_data, mode)

    def _run_and_save_individual_mixed_execution(
        self, operations: List, test_data: Dict[str, Any], validation_dir: str
    ) -> Dict[str, Any]:
        """Run individual mixed operations execution and save results with IDs."""
        logger.info("🔍 Running individual mixed operations execution for validation...")

        # Debug initial test_data
        logger.debug(f"🔍 DEBUG: Initial test_data keys: {list(test_data.keys())}")
        logger.debug(f"🔍 DEBUG: Initial test_data structure: {test_data}")
        if "text" in test_data:
            logger.debug(f"🔍 DEBUG: Initial text field type: {type(test_data['text'])}")
            logger.info(
                f"🔍 DEBUG: Initial text field length: {len(test_data['text']) if hasattr(test_data['text'], '__len__') else 'N/A'}"
            )
            if hasattr(test_data["text"], "__len__") and len(test_data["text"]) > 0:
                logger.debug(f"🔍 DEBUG: Initial first text sample: {test_data['text'][0]}")
                logger.debug(f"🔍 DEBUG: Initial first text sample type: {type(test_data['text'][0])}")

        # Add sample IDs to test data
        original_length = get_dataset_length(test_data)
        test_data_with_ids = test_data.copy()
        test_data_with_ids["sample_id"] = list(range(original_length))

        logger.debug(f"🔍 DEBUG: Starting with {original_length} samples")
        logger.debug(f"🔍 DEBUG: Test data keys: {list(test_data_with_ids.keys())}")
        logger.debug(f"🔍 DEBUG: Text samples: {len(test_data_with_ids.get('text', []))}")

        # Process through all operations
        data = test_data_with_ids
        for i, op in enumerate(operations):
            op_type = type(op).__name__
            logger.info(f"🔍 STEP {i+1}/{len(operations)}: {op_type}")
            logger.info(f"   📊 BEFORE: {len(data.get('text', []))} samples")
            if len(data.get("text", [])) > 0:
                logger.info(f"   📝 First text sample: {str(data['text'][0])[:100]}...")
            logger.info(f"   🔑 Data keys: {list(data.keys())}")

            if hasattr(op, "compute_stats_batched"):
                logger.info(f"   📈 Computing stats for {op_type}...")
                data = op.compute_stats_batched(data)
                logger.info(f"   ✅ Stats computed: {len(data.get('text', []))} samples")

            if hasattr(op, "process_batched"):
                logger.info(f"   🔄 Processing with {op_type}...")
                result = list(op.process_batched(data))
                logger.info(f"   📋 Result type: {type(result[0]) if result else 'None'}")
                logger.info(f"   📏 Result length: {len(result) if result else 0}")

                if result and len(result) > 0:
                    logger.info(f"   🎯 First result: {result[0]}")
                    logger.info(f"   🎯 First result type: {type(result[0])}")

                # Check if this is a filter (returns boolean) or mapper (returns transformed data)
                if result and isinstance(result[0], bool):
                    # This is a filter - apply boolean mask
                    mask = result
                    passed_count = sum(mask)
                    total_count = len(mask)
                    logger.info(
                        f"   🚦 FILTER RESULT: {passed_count}/{total_count} samples passed ({passed_count/total_count*100:.1f}%)"
                    )

                    # Keep only samples that passed the filter
                    passed_indices = [idx for idx, passed in enumerate(mask) if passed]
                    if passed_indices:
                        # Update data to keep only passed samples
                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) == len(mask):
                                data[key] = [data[key][idx] for idx in passed_indices]
                        logger.info(f"   ✅ AFTER FILTER: {len(passed_indices)} samples remaining")
                        if len(passed_indices) > 0:
                            logger.info(f"   📝 First remaining text: {str(data['text'][0])[:100]}...")
                    else:
                        # No samples passed - clear all data
                        for key in data:
                            if isinstance(data[key], list):
                                data[key] = []
                        logger.info(f"   ❌ AFTER FILTER: 0 samples remaining - STOPPING")
                        break
                else:
                    # This is a mapper - update text data
                    if result:
                        logger.info(f"   🔄 MAPPER RESULT: Updating text field")
                        data["text"] = result
                        # Keep sample_ids and stats aligned
                        if "sample_id" in data and len(data["sample_id"]) != len(result):
                            data["sample_id"] = data["sample_id"][: len(result)]
                        if Fields.stats in data and len(data[Fields.stats]) != len(result):
                            data[Fields.stats] = data[Fields.stats][: len(result)]
                        logger.info(f"   ✅ AFTER MAPPER: {len(data.get('text', []))} samples")
                        if len(data.get("text", [])) > 0:
                            logger.info(f"   📝 First text after mapper: {str(data['text'][0])[:100]}...")
                    else:
                        logger.warning(f"   ⚠️ MAPPER RESULT: Empty result from {op_type}")

            logger.info(f"   📊 AFTER STEP {i+1}: {len(data.get('text', []))} samples")
            logger.info(f"   " + "=" * 50)

        logger.debug(f"🔍 DEBUG: Final individual execution: {len(data.get('text', []))} samples")
        logger.debug(f"🔍 DEBUG: Final data keys: {list(data.keys())}")

        # Save individual execution results
        individual_results_file = os.path.join(validation_dir, "individual_execution_results.jsonl")
        self._save_results_to_file(data, individual_results_file)
        logger.info(f"📄 Individual execution results saved to: {individual_results_file}")

        return data

    def _run_and_save_pipeline_mixed_execution(
        self, ast, test_data: Dict[str, Any], validation_dir: str
    ) -> Dict[str, Any]:
        """Run pipeline mixed operations execution and save results with IDs."""
        logger.info("🔍 Running pipeline mixed operations execution for validation...")

        # Debug initial test_data
        logger.debug(f"🔍 DEBUG: Pipeline initial test_data keys: {list(test_data.keys())}")
        logger.debug(f"🔍 DEBUG: Pipeline initial test_data structure: {test_data}")
        if "text" in test_data:
            logger.debug(f"🔍 DEBUG: Pipeline initial text field type: {type(test_data['text'])}")
            logger.info(
                f"🔍 DEBUG: Pipeline initial text field length: {len(test_data['text']) if hasattr(test_data['text'], '__len__') else 'N/A'}"
            )
            if hasattr(test_data["text"], "__len__") and len(test_data["text"]) > 0:
                logger.debug(f"🔍 DEBUG: Pipeline initial first text sample: {test_data['text'][0]}")
                logger.debug(f"🔍 DEBUG: Pipeline initial first text sample type: {type(test_data['text'][0])}")

        # Add sample IDs to test data
        original_length = get_dataset_length(test_data)
        test_data_with_ids = test_data.copy()
        test_data_with_ids["sample_id"] = list(range(original_length))

        logger.debug(f"🔍 DEBUG: Pipeline starting with {original_length} samples")
        logger.debug(f"🔍 DEBUG: Pipeline test data keys: {list(test_data_with_ids.keys())}")
        logger.debug(f"🔍 DEBUG: Pipeline text samples: {len(test_data_with_ids.get('text', []))}")

        # Convert AST back to config for pipeline execution
        config = self._convert_ast_to_config(ast)
        logger.debug(f"🔍 DEBUG: Pipeline config: {config}")

        # Create pipeline optimizer and optimize
        optimizer = PipelineOptimizer(strategies=[FilterFusionStrategy(), MapperFusionStrategy()])

        # Create AST from config
        ast2 = PipelineAST()
        ast2.build_from_config(config)
        logger.debug("🔍 DEBUG: Pipeline AST created.")

        # Optimize AST
        optimized_ast = optimizer.optimize(ast2)
        logger.debug("🔍 DEBUG: Pipeline optimized AST created.")

        # Convert optimized AST back to operations (this handles the double nesting properly)
        optimized_op_configs = self._convert_ast_to_operations(optimized_ast)
        logger.info(f"🔍 DEBUG: Pipeline optimized operation configs: {optimized_op_configs}")
        logger.debug(f"🔍 DEBUG: Pipeline loaded {len(optimized_op_configs)} optimized operation configs")

        # Load the operations from configs
        optimized_ops = []
        for op_config in optimized_op_configs:
            op_name = list(op_config.keys())[0]
            op_args = op_config[op_name]
            if op_name == "fused_filter":
                fused_op_list = op_args.get("fused_op_list", [])
                individual_filters = []
                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]

                    # Skip if this is a nested fused_filter (already optimized)
                    if filter_name == "fused_filter":
                        logger.warning(f"🔍 DEBUG: Skipping nested fused_filter in {op_name}")
                        continue

                    loaded_filters = load_ops([{filter_name: filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])
                if individual_filters:
                    fused_filter = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    fused_filter.execution_strategy = "sequential"
                    optimized_ops.append(fused_filter)
            elif op_name == "fused_mapper":
                mapper_config = op_args
                name = mapper_config.get("name", "fused_mapper")
                fused_mappers = mapper_config.get("fused_mappers", [])
                fused_mapper = FusedMapper(name=name, fused_mappers=fused_mappers)
                optimized_ops.append(fused_mapper)
            else:
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    optimized_ops.append(loaded_ops[0])

        # Process with optimized operations
        data = test_data_with_ids
        logger.info(f"🔍 PIPELINE: Starting with {len(data.get('text', []))} samples")
        logger.info(f"🔍 PIPELINE: Data keys: {list(data.keys())}")

        for i, op in enumerate(optimized_ops):
            op_type = type(op).__name__
            logger.info(f"🔍 PIPELINE STEP {i+1}/{len(optimized_ops)}: {op_type}")
            logger.info(f"   📊 BEFORE: {len(data.get('text', []))} samples")
            if len(data.get("text", [])) > 0:
                logger.info(f"   📝 First text sample: {str(data['text'][0])[:100]}...")
            logger.info(f"   🔑 Data keys: {list(data.keys())}")

            if hasattr(op, "compute_stats_batched"):
                logger.info(f"   📈 Computing stats for {op_type}...")
                data = op.compute_stats_batched(data)
                logger.info(f"   ✅ Stats computed: {len(data.get('text', []))} samples")

            if hasattr(op, "process_batched"):
                logger.info(f"   🔄 Processing with {op_type}...")
                result = list(op.process_batched(data))
                logger.info(f"   📋 Result type: {type(result[0]) if result else 'None'}")
                logger.info(f"   📏 Result length: {len(result) if result else 0}")

                if result and len(result) > 0:
                    logger.info(f"   🎯 First result: {result[0]}")
                    logger.info(f"   🎯 First result type: {type(result[0])}")

                # Check if this is a filter (returns boolean) or mapper (returns transformed data)
                if result and isinstance(result[0], bool):
                    # This is a filter - apply boolean mask
                    mask = result
                    passed_count = sum(mask)
                    total_count = len(mask)
                    logger.info(
                        f"   🚦 PIPELINE FILTER RESULT: {passed_count}/{total_count} samples passed ({passed_count/total_count*100:.1f}%)"
                    )

                    # Keep only samples that passed the filter
                    passed_indices = [idx for idx, passed in enumerate(mask) if passed]
                    if passed_indices:
                        # Update data to keep only passed samples
                        for key in data:
                            if isinstance(data[key], list) and len(data[key]) == len(mask):
                                data[key] = [data[key][idx] for idx in passed_indices]
                        logger.info(f"   ✅ PIPELINE AFTER FILTER: {len(passed_indices)} samples remaining")
                        if len(passed_indices) > 0:
                            logger.info(f"   📝 First remaining text: {str(data['text'][0])[:100]}...")
                    else:
                        # No samples passed - clear all data
                        for key in data:
                            if isinstance(data[key], list):
                                data[key] = []
                        logger.info(f"   ❌ PIPELINE AFTER FILTER: 0 samples remaining - STOPPING")
                        break
                else:
                    # This is a mapper - update text data
                    if result:
                        logger.info(f"   🔄 PIPELINE MAPPER RESULT: Updating text field")
                        data["text"] = result
                        # Keep sample_ids and stats aligned
                        if "sample_id" in data and len(data["sample_id"]) != len(result):
                            data["sample_id"] = data["sample_id"][: len(result)]
                        if Fields.stats in data and len(data[Fields.stats]) != len(result):
                            data[Fields.stats] = data[Fields.stats][: len(result)]
                        logger.info(f"   ✅ PIPELINE AFTER MAPPER: {len(data.get('text', []))} samples")
                        if len(data.get("text", [])) > 0:
                            logger.info(f"   📝 First text after mapper: {str(data['text'][0])[:100]}...")
                    else:
                        logger.warning(f"   ⚠️ PIPELINE MAPPER RESULT: Empty result from {op_type}")

            logger.info(f"   📊 PIPELINE AFTER STEP {i+1}: {len(data.get('text', []))} samples")
            logger.info(f"   " + "=" * 50)

        logger.info(f"🔍 PIPELINE FINAL: {len(data.get('text', []))} samples")
        logger.info(f"🔍 PIPELINE FINAL DATA KEYS: {list(data.keys())}")

        # Save pipeline execution results
        pipeline_results_file = os.path.join(validation_dir, "pipeline_execution_results.jsonl")
        self._save_results_to_file(data, pipeline_results_file)
        logger.info(f"📄 Pipeline execution results saved to: {pipeline_results_file}")

        return data

    def run_individual_mixed_ops_benchmark(self, operations: List, test_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark individual mixed operations executed sequentially."""
        logger.info("Running individual mixed operations benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Step 1: Initialize operations
        init_start = time.time()
        # Filter out operations that don't have compute_stats_batched
        actual_ops = [op for op in operations if hasattr(op, "compute_stats_batched")]
        logger.info(
            f"  Found {len(actual_ops)} operations with compute_stats_batched out of {len(operations)} operations"
        )

        if not actual_ops:
            logger.warning("  No operations found to benchmark!")
            return PerformanceMetrics(
                total_time=0.0,
                stats_time=0.0,
                filter_time=0.0,
                memory_usage=0.0,
                throughput=0.0,
            )

        init_time = time.time() - init_start
        logger.info(f"  Step 1 - Operation initialization: {init_time:.3f}s")

        # Step 2: Process each operation completely (stats + processing) before moving to the next
        processing_start = time.time()
        samples_with_stats = test_data
        total_stats_time = 0.0
        total_processing_time = 0.0

        for i, op in enumerate(actual_ops):
            op_type = type(op).__name__
            logger.info(f"    Processing operation {i+1}/{len(actual_ops)}: {op_type}")

            # Compute stats for this operation
            stats_start = time.time()
            if hasattr(op, "compute_stats_batched"):
                samples_with_stats = op.compute_stats_batched(samples_with_stats)
            stats_time = time.time() - stats_start
            total_stats_time += stats_time

            # Process with this operation
            processing_start_op = time.time()
            if hasattr(op, "process_batched"):
                result = list(op.process_batched(samples_with_stats))
                # Update data for next operation if this is a mapper
                if "mapper" in op_type.lower() and result:
                    samples_with_stats = {"text": result, "__dj__stats__": [{} for _ in range(len(result))]}
            processing_time = time.time() - processing_start_op
            total_processing_time += processing_time

            logger.debug(f"      Operation {i+1} - Stats: {stats_time:.3f}s, Processing: {processing_time:.3f}s")

        processing_time = time.time() - processing_start
        logger.info(f"  Step 2 - Complete processing: {processing_time:.3f}s")
        logger.info(f"    Total stats time: {total_stats_time:.3f}s")
        logger.info(f"    Total processing time: {total_processing_time:.3f}s")

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = get_dataset_length(test_data) / total_time

        logger.info("  📊 INDIVIDUAL MIXED OPERATIONS BREAKDOWN:")
        logger.info(f"    Initialization: {init_time:.3f}s ({init_time/total_time*100:.1f}%)")
        logger.info(f"    Stats computation: {total_stats_time:.3f}s ({total_stats_time/total_time*100:.1f}%)")
        logger.info(f"    Processing: {total_processing_time:.3f}s ({total_processing_time/total_time*100:.1f}%)")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_stats_time,
            filter_time=total_processing_time,  # Rename this field in the future
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def run_pipeline_mixed_ops_benchmark(
        self, operations: List, test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Benchmark the complete pipeline optimizer workflow for mixed operations."""
        logger.info("Running pipeline mixed operations benchmark (complete workflow)...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Step 1: Build pipeline configuration from operations
        pipeline_config = self._build_pipeline_config_from_operations(operations)

        # Step 2: Create Pipeline AST
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        # Step 3: Create PipelineOptimizer with fusion strategies
        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )

        strategies = [FilterFusionStrategy(), MapperFusionStrategy()]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)

        # Step 4: Get optimization summary
        optimization_summary = optimizer.get_optimization_summary()
        logger.info("  Pipeline Optimizer Configuration:")
        logger.info(f"    Strategies: {optimization_summary['strategies']}")
        logger.info(f"    Analyzer insights: {optimization_summary['analyzer_insights_available']}")

        # Step 5: Apply optimizations
        logger.info("  Applying pipeline optimizations...")
        optimized_ast = optimizer.optimize(ast)

        # Step 6: Convert optimized AST back to operations
        optimized_ops = self._convert_ast_to_operations(optimized_ast)
        logger.info(f"  Original operations: {len(operations)}")
        logger.info(f"  Optimized operations: {len(optimized_ops)}")

        # Step 7: Process with optimized operations
        logger.info("  Processing with optimized pipeline...")
        self._process_with_optimized_mixed_ops(optimized_ops, test_data)

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = get_dataset_length(test_data) / total_time

        logger.info("  📊 PIPELINE MIXED OPERATIONS BREAKDOWN:")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")
        logger.info(f"    Memory usage: {memory_usage:.1f} MB")
        logger.info(f"    Optimization ratio: {len(optimized_ops)/len(operations):.2f}x")
        logger.info(f"    Operations reduced: {len(operations) - len(optimized_ops)}")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.8,  # Estimate: most time is processing
            filter_time=total_time * 0.2,  # Estimate: some time is optimization
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def _build_pipeline_config_from_operations(self, operations: List) -> Dict[str, Any]:
        """Build pipeline config from mixed operations."""
        process_config = []
        for i, op in enumerate(operations):
            op_name = getattr(op, "_name", f"operation_{i}")
            op_config = getattr(op, "config", None)
            if op_config:
                process_config.append({op_name: op_config})
            else:
                # Create basic config from operation attributes
                config_dict = {}
                for attr in dir(op):
                    if not attr.startswith("_") and not callable(getattr(op, attr)):
                        value = getattr(op, attr)
                        if isinstance(value, (int, float, str, bool)):
                            config_dict[attr] = value
                process_config.append({op_name: config_dict})
        return {"process": process_config}

    def _process_with_optimized_mixed_ops(self, optimized_ops: List, test_data: Dict[str, Any]):
        """Process test data with optimized mixed operations."""
        logger.debug(f"Processing with {len(optimized_ops)} optimized operations")

        # Load and execute the optimized operations
        from data_juicer.ops import load_ops

        data = test_data
        for op_config in optimized_ops:
            # Debug: Log the operation configuration
            logger.debug(f"Loading operation config: {op_config}")

            # Special handling for fused operations
            op_name = list(op_config.keys())[0]
            if op_name == "fused_filter":
                # Handle fused filter
                fused_op_list = op_config[op_name].get("fused_op_list", [])
                individual_filters = []

                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]

                    # Skip if this is a nested fused_filter (already optimized)
                    if filter_name == "fused_filter":
                        logger.warning(f"🔍 DEBUG: Skipping nested fused_filter in {op_name}")
                        continue

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

                    loaded_filters = load_ops([{filter_name: clean_filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])

                if individual_filters:
                    fused_filter = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force parallel execution to match individual execution behavior
                    # (each filter sees original data, not filtered output from previous filters)
                    fused_filter.execution_strategy = "parallel"

                    if hasattr(fused_filter, "compute_stats_batched"):
                        data = fused_filter.compute_stats_batched(data)
                    if hasattr(fused_filter, "process_batched"):
                        result = list(fused_filter.process_batched(data))
                        # Update data for next operation
                        if result and isinstance(result[0], bool):
                            # This is a filter - apply boolean mask
                            pass  # Keep original data structure
                        else:
                            # This is a mapper - update data
                            data = {"text": result, "__dj__stats__": [{} for _ in range(len(result))]}

            elif op_name == "fused_mapper":
                # Handle fused mapper
                mapper_config = op_config[op_name]
                name = mapper_config.get("name", "fused_mapper")
                fused_mappers = mapper_config.get("fused_mappers", [])

                fused_mapper = FusedMapper(name=name, fused_mappers=fused_mappers)

                if hasattr(fused_mapper, "compute_stats_batched"):
                    data = fused_mapper.compute_stats_batched(data)
                if hasattr(fused_mapper, "process_batched"):
                    result = list(fused_mapper.process_batched(data))
                    # Mappers always return transformed data
                    if result:
                        data = {"text": result, "__dj__stats__": [{} for _ in range(len(result))]}

            else:
                # Load the operation from config for non-fused operations
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    op = loaded_ops[0]

                    # Execute the operation
                    if hasattr(op, "compute_stats_batched"):
                        data = op.compute_stats_batched(data)
                    if hasattr(op, "process_batched"):
                        result = list(op.process_batched(data))
                        # Check if this is a mapper or filter
                        if result and isinstance(result[0], bool):
                            # This is a filter - keep original data structure
                            pass
                        else:
                            # This is a mapper - update data
                            if result:
                                data = {"text": result, "__dj__stats__": [{} for _ in range(len(result))]}


def create_simple_test_data(num_samples: int = 1000) -> Dict[str, Any]:
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
    return {"text": texts, Fields.stats: [{} for _ in range(num_samples)]}


def main():
    """Main execution function for performance benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Benchmark for Data-Juicer Filters")
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

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = PerformanceBenchmark()

    # Create test data
    if args.dataset_path:
        logger.info(f"🎯 Using real dataset: {args.dataset_path}")
        test_data = load_real_dataset(args.dataset_path, max_samples=args.samples)
    else:
        logger.info(f"🎲 Using synthetic data: {args.samples} samples")
        test_data = benchmark.create_realistic_test_data(args.samples)

    # Get filters based on mode
    if args.mode == "quick":
        filters = benchmark.create_quick_test_filters()
    elif args.mode == "full":
        filters = benchmark.create_full_test_filters()  # Use comprehensive test filters (12 filters)
    elif args.mode == "recipe":
        # Load pipeline from YAML recipe using optimizer framework
        if not args.recipe_path:
            raise ValueError("--recipe-path must be specified in recipe mode!")

        # Build AST from recipe
        ast = PipelineAST()
        ast.build_from_yaml(args.recipe_path)
        logger.info(f"📋 Loaded recipe: {args.recipe_path}")
        logger.info(f"Pipeline structure:\n{ast.visualize()}")

        # Compute analyzer insights for the test data
        analyzer_insights = benchmark.get_analyzer_insights(test_data)

        # Use optimizer to handle the full pipeline, passing analyzer_insights only to PipelineOptimizer
        optimizer = PipelineOptimizer(
            [FilterFusionStrategy(), MapperFusionStrategy()], analyzer_insights=analyzer_insights
        )

        # Optimize the pipeline
        optimized_ast = optimizer.optimize(ast)
        logger.info(f"🔧 Optimized pipeline structure:\n{optimized_ast.visualize()}")

        # Convert original AST to operations for individual execution
        original_operations = benchmark._convert_ast_to_operations(ast)
        logger.info(f"🔍 Extracted {len(original_operations)} original operations from AST:")
        for i, op in enumerate(original_operations):
            logger.info(f"  {i+1}: {op}")

        # Convert optimized AST to operations for pipeline execution
        optimized_operations = benchmark._convert_ast_to_operations(optimized_ast)
        logger.info(f"🔍 Extracted {len(optimized_operations)} optimized operations from optimized AST:")
        for i, op in enumerate(optimized_operations):
            logger.info(f"  {i+1}: {op}")

        # Load original operations for individual execution
        from data_juicer.ops.load import load_ops

        loaded_original_operations = []
        for op_config in original_operations:
            op_name = list(op_config.keys())[0]
            op_args = op_config[op_name]
            loaded_ops = load_ops([op_config])
            if loaded_ops:
                loaded_original_operations.append(loaded_ops[0])

        logger.info(f"📊 Loaded {len(loaded_original_operations)} original operations for individual execution")

        # Load optimized operations for pipeline execution
        loaded_optimized_operations = []
        for op_config in optimized_operations:
            op_name = list(op_config.keys())[0]
            op_args = op_config[op_name]
            if op_name == "fused_filter":
                fused_op_list = op_args.get("fused_op_list", [])
                individual_filters = []
                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]
                    loaded_filters = load_ops([{filter_name: filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])
                if individual_filters:
                    fused_filter = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force sequential execution to avoid parallel stats access issues
                    fused_filter.execution_strategy = "sequential"
                    loaded_optimized_operations.append(fused_filter)
            elif op_name == "fused_mapper":
                # Handle fused mapper
                mapper_config = op_args
                name = mapper_config.get("name", "fused_mapper")
                fused_mappers = mapper_config.get("fused_mappers", [])

                fused_mapper = FusedMapper(name=name, fused_mappers=fused_mappers)
                loaded_optimized_operations.append(fused_mapper)
            else:
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    loaded_optimized_operations.append(loaded_ops[0])

        logger.info(f"📊 Loaded {len(loaded_optimized_operations)} optimized operations for pipeline execution")

        # Use mixed operations benchmarking for recipe mode
        # This properly handles mappers, filters, and other operation types
        results = benchmark.run_mixed_operations_benchmark_with_original_ops(
            loaded_original_operations, loaded_optimized_operations, test_data, args.mode
        )

        logger.info("\n✅ Recipe benchmark completed successfully!")
        logger.info(f"📊 Results saved for mode: {args.mode}")

        return results
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Get analyzer insights if requested
    analyzer_insights = None
    if args.analyzer:
        logger.info("🔍 Running analyzer to get insights...")
        analyzer_insights = benchmark.run_analyzer(test_data)

    # Run comprehensive benchmark with fusion strategies
    results = benchmark.run_benchmark(filters, test_data, args.mode, analyzer_insights)

    logger.info("\n✅ Benchmark completed successfully!")
    logger.info(f"📊 Results saved for mode: {args.mode}")

    return results


def load_real_dataset(dataset_path: str, max_samples: int = None) -> Dict[str, Any]:
    """
    Load real dataset using DatasetBuilder and convert to expected format.

    Args:
        dataset_path: Path to the dataset file
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Dictionary with 'text' and Fields.stats keys for benchmark compatibility
    """
    from argparse import Namespace

    from data_juicer.core.data.dataset_builder import DatasetBuilder
    from data_juicer.utils.constant import Fields

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
    if max_samples is not None and hasattr(dataset, "__len__") and len(dataset) > max_samples:
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(max_samples))
        logger.info(f"✅ Limited to {max_samples} samples from {dataset_path}")
    else:
        logger.info(f"✅ Loaded {len(dataset)} samples from {dataset_path}")

    # Log dataset info
    if hasattr(dataset, "column_names"):
        logger.info(f"📊 Dataset columns: {dataset.column_names}")
        if "text" in dataset.column_names:
            if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
                sample_texts = dataset["text"][:3] if len(dataset) >= 3 else dataset["text"]
                avg_length = sum(len(str(t)) for t in sample_texts) / len(sample_texts) if sample_texts else 0
                logger.info(f"📝 Sample text lengths: avg={avg_length:.1f} chars")

    # Convert to expected format for benchmark
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        texts = dataset["text"] if "text" in dataset.column_names else []
        # Create stats list with empty dicts for each sample
        stats = [{} for _ in range(len(texts))]
        return {"text": texts, Fields.stats: stats}
    else:
        # Fallback: return empty dataset
        return {"text": [], Fields.stats: []}


if __name__ == "__main__":
    main()
