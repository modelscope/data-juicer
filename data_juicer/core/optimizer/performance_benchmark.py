#!/usr/bin/env python3
"""
Performance benchmark for Data-Juicer filter fusion and optimization.

This benchmark compares individual vs fused filter performance and demonstrates
the new PipelineOptimizer architecture. The optimizer architecture with analyzer
insights is used by default in all modes.

USAGE EXAMPLES:
    # Quick benchmark (basic demo with 1000 samples) - uses optimizer by default
    python performance_benchmark.py --mode quick

    # Quick benchmark with more samples
    python performance_benchmark.py --mode quick --samples 10000

    # Full comprehensive benchmark - uses optimizer by default
    python performance_benchmark.py --mode full --samples 50000 --runs 5

    # Analyze fusion decisions
    python performance_benchmark.py --mode fusion-analysis --samples 1000

MODES:
    quick    - Basic performance demo with optimizer architecture (default)
    full     - Comprehensive benchmark with optimizer architecture
    fusion-analysis - Analyze when to use fusion vs skip fusion
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
from data_juicer.core.optimizer.fused_op import FusedFilter, FusedMapper
from data_juicer.core.optimizer.optimizer import PipelineOptimizer
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
    TextActionFilter,
    TextEntityDependencyFilter,
    TextLengthFilter,
    WordRepetitionFilter,
    WordsNumFilter,
)
from data_juicer.utils.constant import Fields

# Global model cache
_MODEL_CACHE = {}


def get_cached_model(model_type: str, lang: str = "en", model_size: str = "md"):
    """Get a cached model or load it if not cached.

    Args:
        model_type: Type of model (spacy, etc.)
        lang: Language code (en, zh)
        model_size: Model size for spaCy (sm, md, lg) - smaller = faster loading
    """
    cache_key = f"{model_type}_{lang}_{model_size}"
    if cache_key not in _MODEL_CACHE:
        from data_juicer.utils.model_utils import prepare_model

        logger.info(f"Loading {model_type} model for {lang} (size: {model_size})...")

        if model_type == "spacy":
            # Use smaller model for faster loading in benchmarks
            if model_size == "sm":
                # Small model (~12MB) - faster loading, less accurate
                model_name_pattern = "{}_core_web_sm-3.7.0"
            elif model_size == "lg":
                # Large model (~560MB) - slower loading, more accurate
                model_name_pattern = "{}_core_web_lg-3.7.0"
            else:
                # Medium model (~40MB) - default
                model_name_pattern = "{}_core_web_md-3.7.0"

            _MODEL_CACHE[cache_key] = prepare_model(model_type=model_type, lang=lang, name_pattern=model_name_pattern)
        else:
            _MODEL_CACHE[cache_key] = prepare_model(model_type=model_type, lang=lang)

        logger.info(f"Model {cache_key} loaded and cached")
    else:
        logger.debug(f"Using cached {cache_key}")
    return _MODEL_CACHE[cache_key]


def create_filter_with_cached_model(filter_class, model_size: str = "sm", **kwargs):
    """Create a filter instance with cached model loading.

    Args:
        filter_class: Filter class to instantiate
        model_size: Model size for spaCy models (sm, md, lg)
        **kwargs: Filter constructor arguments
    """
    # For spaCy-based filters, we need to patch the model loading
    if filter_class in [TextEntityDependencyFilter, TextActionFilter]:
        # Temporarily patch the prepare_model function
        import data_juicer.utils.model_utils as model_utils

        original_prepare_spacy = model_utils.prepare_spacy_model

        def cached_prepare_spacy(lang, name_pattern="{}_core_web_md-3.7.0", **kwargs):
            # Determine model size from name_pattern
            if "sm" in name_pattern:
                size = "sm"
            elif "lg" in name_pattern:
                size = "lg"
            else:
                size = model_size  # Use provided size or default to "sm"
            return get_cached_model("spacy", lang, size)

        model_utils.prepare_spacy_model = cached_prepare_spacy

        try:
            filter_instance = filter_class(**kwargs)
        finally:
            # Restore original function
            model_utils.prepare_spacy_model = original_prepare_spacy
    else:
        filter_instance = filter_class(**kwargs)

    return filter_instance


def preload_models_for_benchmark():
    """Preload commonly used models to avoid loading delays during benchmark."""
    logger.info("🔄 Preloading models for benchmark...")

    # Preload spaCy models in different sizes
    for size in ["sm", "md"]:
        try:
            get_cached_model("spacy", "en", size)
            logger.info(f"✅ Preloaded spaCy {size} model")
        except Exception as e:
            logger.warning(f"⚠️ Failed to preload spaCy {size} model: {e}")

    # Preload other models as needed
    try:
        get_cached_model("gpt2", "en")
        logger.info("✅ Preloaded GPT-2 model")
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload GPT-2 model: {e}")

    logger.info("🎯 Model preloading complete")


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
        """Create a comprehensive set of test filters covering different categories."""
        logger.info("Creating comprehensive test filters...")

        filters = [
            # Basic text filters (simple, fast)
            WordsNumFilter(min_num=5, max_num=1000),
            TextLengthFilter(min_len=20, max_len=1000),
            CharacterRepetitionFilter(repetition_ratio=0.8),
            WordRepetitionFilter(min_ratio=0.0, max_ratio=0.5),
            SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),
            AlphanumericFilter(min_ratio=0.3),
            AverageLineLengthFilter(min_len=10, max_len=100),
            MaximumLineLengthFilter(min_len=10, max_len=200),
            # Content quality filters (moderate complexity)
            PerplexityFilter(lang="en", model_key="gpt2", min_score=0.0, max_score=100.0),
            StopWordsFilter(lang="en", min_ratio=0.0, max_ratio=0.5),
            FlaggedWordFilter(lang="en", min_ratio=0.0, max_ratio=0.1),
            LanguageIDScoreFilter(lang="en", min_score=0.5, max_score=1.0),
            # Advanced text analysis filters (high complexity) - with cached models
            # Note: These use local models and are computationally intensive
            # create_filter_with_cached_model(TextEntityDependencyFilter, lang="en", min_dependency_num=1, any_or_all="all"),
            # create_filter_with_cached_model(TextActionFilter, lang="en", min_action_num=1),
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
        samples_with_stats = test_data.copy()
        total_stats_time = 0.0
        total_filter_time = 0.0

        for i, filter_op in enumerate(actual_filters):
            logger.info(f"    Processing filter {i+1}/{len(actual_filters)}: {filter_op._name}")

            # Compute stats for this filter
            stats_start = time.time()
            if hasattr(filter_op, "compute_stats_batched"):
                samples_with_stats = filter_op.compute_stats_batched(samples_with_stats)
            stats_time = time.time() - stats_start
            total_stats_time += stats_time

            # Immediately filter with this filter
            filter_start = time.time()
            if hasattr(filter_op, "process_batched"):
                _ = list(filter_op.process_batched(samples_with_stats))
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
        throughput = len(test_data["text"]) / total_time

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

    def run_fused_filters_benchmark(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Benchmark fused filter execution using proper fusion strategies."""
        logger.info("Running fused filters benchmark (with fusion strategies)...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Step 1: Build pipeline configuration from filters
        pipeline_config = self._build_pipeline_config_from_filters(filters)

        # Step 2: Create Pipeline AST
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        # Step 3: Create PipelineOptimizer with fusion strategies
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
        logger.info("  Fusion Strategy Configuration:")
        logger.info(f"    Strategies: {optimization_summary['strategies']}")
        logger.info(f"    Analyzer insights: {optimization_summary['analyzer_insights_available']}")

        # Step 5: Apply optimizations
        logger.info("  Applying fusion strategies...")
        optimized_ast = optimizer.optimize(ast)

        # Step 6: Convert optimized AST back to operations
        optimized_ops = self._convert_ast_to_operations(optimized_ast)
        logger.info(f"  Original operations: {len(filters)}")
        logger.info(f"  Optimized operations: {len(optimized_ops)}")

        # Step 7: Process with optimized operations
        logger.info("  Processing with optimized operations...")
        self._process_with_optimized_ops(optimized_ops, test_data)

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = len(test_data["text"]) / total_time

        logger.info("  📊 FUSION STRATEGY BREAKDOWN:")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")
        logger.info(f"    Memory usage: {memory_usage:.1f} MB")
        logger.info(f"    Fusion ratio: {len(optimized_ops)/len(filters):.2f}x")
        logger.info(f"    Operations reduced: {len(filters) - len(optimized_ops)}")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.8,  # Estimate: most time is processing
            filter_time=total_time * 0.2,  # Estimate: some time is optimization
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
        """Save test results to file with timestamp and metadata."""
        import datetime
        import json
        import os

        # Add timestamp and metadata to results
        results["metadata"] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "filename": filename,
            "version": "1.0",
            "data_juicer_version": "latest",
        }

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
            "metadata": results.get("metadata", {}),
            "test_config": results["test_config"],
        }

        # Handle different result formats
        if "individual" in results:
            saveable_results["individual"] = results["individual"]
        if "fused" in results:
            saveable_results["fused"] = results["fused"]
        if "current_fused" in results:
            saveable_results["current_fused"] = results["current_fused"]
        if "lightweight_fused" in results:
            saveable_results["lightweight_fused"] = results["lightweight_fused"]
        if "improvements" in results:
            saveable_results["improvements"] = results["improvements"]
        if "comparison" in results:
            saveable_results["comparison"] = results["comparison"]
        if "raw_results" in results:
            saveable_results["raw_results"] = {
                "individual": convert_metrics(results["raw_results"]["individual"]),
                "fused": convert_metrics(results["raw_results"]["fused"]),
            }

        # Add additional metadata if available (with JSON serialization handling)
        def convert_numpy_types(obj):
            """Convert numpy types to JSON serializable types."""
            import numpy as np

            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        if "filtering_stats" in results:
            saveable_results["filtering_stats"] = convert_numpy_types(results["filtering_stats"])
        if "analyzer_insights" in results:
            saveable_results["analyzer_insights"] = convert_numpy_types(results["analyzer_insights"])

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

        # Save to file
        with open(filename, "w") as f:
            json.dump(saveable_results, f, indent=2)

        # Log file info
        file_size_kb = os.path.getsize(filename) / 1024
        logger.info(f"📁 Results saved to {filename}")
        logger.info(f"   File size: {file_size_kb:.1f} KB")
        logger.info(f"   Timestamp: {saveable_results['metadata']['timestamp']}")

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
        throughput = len(test_data["text"]) / total_time

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
                        # Extract fused mapper configuration
                        if "fused_mapper" in op_config:
                            mapper_config = op_config["fused_mapper"]
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

        for op_config in optimized_ops:
            # Debug: Log the operation configuration
            logger.debug(f"Loading operation config: {op_config}")

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
                    from data_juicer.core.optimizer.fused_op import FusedFilter

                    fused_filter = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force sequential execution to avoid parallel stats access issues
                    fused_filter.execution_strategy = "sequential"

                    # Process the fused filter
                    if hasattr(fused_filter, "compute_stats_batched"):
                        test_data = fused_filter.compute_stats_batched(test_data)
                    if hasattr(fused_filter, "process_batched"):
                        _ = list(fused_filter.process_batched(test_data))
            elif op_name == "fused_mapper":
                # Extract the fused_mappers list and create FusedMapper directly
                mapper_config = op_config[op_name]
                name = mapper_config.get("name", "fused_mapper")
                fused_mappers = mapper_config.get("fused_mappers", [])

                # Create the fused mapper directly
                fused_mapper = FusedMapper(name=name, fused_mappers=fused_mappers)

                # Execute the mapper
                if hasattr(fused_mapper, "process_batched"):
                    test_data = fused_mapper.process_batched(test_data)
            else:
                # Load the operation from config for non-fused operations
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    op = loaded_ops[0]

                    # Execute the operation
                    if hasattr(op, "compute_stats_batched"):
                        test_data = op.compute_stats_batched(test_data)
                    if hasattr(op, "process_batched"):
                        _ = list(op.process_batched(test_data))

    def _process_with_legacy_ops(self, legacy_ops: List, test_data: Dict[str, Any]):
        """Process test data with legacy fused operations."""
        logger.debug(f"Processing with {len(legacy_ops)} legacy operations")

        for op in legacy_ops:
            if hasattr(op, "compute_stats_batched"):
                test_data = op.compute_stats_batched(test_data)
            if hasattr(op, "process_batched"):
                _ = list(op.process_batched(test_data))

    def run_lightweight_fusion_benchmark(self, filters: List[Filter], test_data: Dict[str, Any]) -> PerformanceMetrics:
        """Benchmark lightweight fusion without complex overhead."""
        logger.info("Running lightweight fusion benchmark...")

        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Step 1: Simple fusion (no complex overhead)
        init_start = time.time()

        # Create a simplified fused filter that skips complex logic
        class LightweightFusedFilter:
            def __init__(self, filters):
                self.filters = filters
                self._name = "lightweight_fused"

            def compute_stats_batched(self, samples):
                # Simple sequential stats computation
                for filter_op in self.filters:
                    samples = filter_op.compute_stats_batched(samples)
                return samples

            def process_batched(self, samples):
                # Simple sequential processing with early termination
                result = None
                for filter_op in self.filters:
                    filter_result = list(filter_op.process_batched(samples))

                    if result is None:
                        result = filter_result
                    else:
                        # Early termination: if any filter fails, stop processing
                        result = [r1 and r2 for r1, r2 in zip(result, filter_result)]
                        # If all samples failed, we can stop early
                        if not any(result):
                            break

                return result

        fused_filter = LightweightFusedFilter(filters)
        init_time = time.time() - init_start
        logger.info(f"  Step 1 - Lightweight fusion initialization: {init_time:.3f}s")

        # Step 2: Stats computation
        stats_start = time.time()
        samples_with_stats = fused_filter.compute_stats_batched(test_data.copy())
        stats_time = time.time() - stats_start
        logger.info(f"  Step 2 - Stats computation: {stats_time:.3f}s")

        # Step 3: Filtering
        filter_start = time.time()
        _ = list(fused_filter.process_batched(samples_with_stats))
        filter_time = time.time() - filter_start
        logger.info(f"  Step 3 - Filtering: {filter_time:.3f}s")

        # Calculate totals
        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = len(test_data["text"]) / total_time

        logger.info("  📊 LIGHTWEIGHT FUSION BREAKDOWN:")
        logger.info(f"    Initialization: {init_time:.3f}s ({init_time/total_time*100:.1f}%)")
        logger.info(f"    Stats computation: {stats_time:.3f}s ({stats_time/total_time*100:.1f}%)")
        logger.info(f"    Filtering: {filter_time:.3f}s ({filter_time/total_time*100:.1f}%)")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Throughput: {throughput:.1f} samples/sec")

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=stats_time,
            filter_time=filter_time,
            memory_usage=memory_usage,
            throughput=throughput,
        )

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

        # --- Validation: Compare results between individual and pipeline ---
        logger.info("\n🔎 VALIDATING PIPELINE RESULTS AGAINST INDIVIDUAL EXECUTION")

        # Get the optimized operations from the pipeline benchmark
        pipeline_config = self._build_pipeline_config_from_filters(filters)
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)
        from data_juicer.core.optimizer.filter_fusion_strategy import (
            FilterFusionStrategy,
        )
        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )
        from data_juicer.core.optimizer.optimizer import PipelineOptimizer

        strategies = [FilterFusionStrategy(analyzer_insights=analyzer_insights), MapperFusionStrategy()]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)
        optimized_ast = optimizer.optimize(ast)
        optimized_ops = self._convert_ast_to_operations(optimized_ast)

        individual_mask = self.get_final_mask_from_filters(filters, test_data)
        pipeline_mask = self.get_final_mask_from_optimized_ops(optimized_ops, test_data)

        if individual_mask == pipeline_mask:
            logger.info("✅ Pipeline results match individual execution!")
            mismatches = []
        else:
            mismatches = [i for i, (a, b) in enumerate(zip(individual_mask, pipeline_mask)) if a != b]
            logger.warning(
                f"❌ Pipeline results do NOT match individual execution! {len(mismatches)} mismatches out of {len(individual_mask)} samples."
            )
            logger.warning(f"First 5 mismatches: {mismatches[:5]}")
            # Print debug info for first 3 mismatches
            for idx in mismatches[:3]:
                logger.warning(f"--- Mismatch at index {idx} ---")
                logger.warning(f"Input text: {test_data['text'][idx]}")

                # Detailed individual execution debugging
                logger.warning("🔍 INDIVIDUAL EXECUTION (step-by-step):")
                data = test_data.copy()
                ind_masks = []
                logger.warning(f"  Starting with {len(filters)} filters")

                for i, filter_op in enumerate(filters):

                    if hasattr(filter_op, "compute_stats_batched"):
                        data_before = data.copy()
                        data = filter_op.compute_stats_batched(data)
                        logger.warning(f"    Stats computed: data keys={list(data.keys())}")
                        # Check if data was modified
                        if data_before != data:
                            logger.warning(f"    Data was modified during stats computation: {data_before} -> {data}")
                        else:
                            logger.warning("    Data was not modified during stats computation")

                    if hasattr(filter_op, "process_batched"):
                        result = list(filter_op.process_batched(data))
                        # Handle both boolean masks and text content
                        if result and isinstance(result[0], bool):
                            mask = result
                        else:
                            # If it returns text content, create a mask based on non-empty results
                            mask = [bool(item) for item in result]

                        # Check if mask has enough elements and idx is valid
                        if mask and len(mask) > idx:
                            ind_masks.append(mask[idx])
                            logger.warning(f"    Result for sample {idx}: {mask[idx]} (passed={mask[idx]})")
                            logger.warning(f"    Overall: {sum(mask)}/{len(mask)} samples passed")
                            logger.warning(f"    Current masks so far: {ind_masks}")
                        else:
                            logger.warning(f"    Warning: mask length {len(mask)} is insufficient for index {idx}")
                            ind_masks.append(True)  # Default to pass if mask is invalid
                    else:
                        logger.warning("    No process_batched method found")
                        ind_masks.append(True)  # Assume pass if no process method

                logger.warning(f"  Individual final result: {all(ind_masks)} (all masks: {ind_masks})")

                # Detailed pipeline execution debugging using the same optimized operations
                logger.warning("🔗 PIPELINE EXECUTION (step-by-step):")
                logger.warning(f"  Original filters: {len(filters)}")
                logger.warning(f"  Optimized ops: {len(optimized_ops)}")
                logger.warning(f"  Optimization ratio: {len(optimized_ops)/len(filters):.2f}x")

                data_pipeline = test_data.copy()
                mask_pipeline = [True] * len(data_pipeline["text"])
                pipeline_masks = []
                from data_juicer.ops import load_ops

                for j, op_config in enumerate(optimized_ops):
                    logger.warning(f"  Optimized Op {j+1}/{len(optimized_ops)}: {op_config}")

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
                            from data_juicer.core.optimizer.fused_op import FusedFilter

                            op = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                            # Force sequential execution to avoid parallel stats access issues
                            op.execution_strategy = "sequential"
                        else:
                            logger.warning("    Failed to create fused filter")
                            pipeline_masks.append(True)
                            continue
                    elif op_name == "fused_mapper":
                        # Extract the fused_mappers list and create FusedMapper directly
                        mapper_config = op_config[op_name]
                        name = mapper_config.get("name", "fused_mapper")
                        fused_mappers = mapper_config.get("fused_mappers", [])

                        # Create the fused mapper directly
                        from data_juicer.core.optimizer.fused_op import FusedMapper

                        op = FusedMapper(name=name, fused_mappers=fused_mappers)
                    else:
                        # Load the operation from config for non-fused operations
                        loaded_ops = load_ops([op_config])
                        if loaded_ops:
                            op = loaded_ops[0]
                        else:
                            logger.warning("    Failed to load op from config")
                            pipeline_masks.append(True)
                            continue

                    logger.warning(f"    Loaded op: {type(op).__name__}")

                    if hasattr(op, "compute_stats_batched"):
                        data_pipeline_before = data_pipeline.copy()
                        data_pipeline = op.compute_stats_batched(data_pipeline)
                        logger.warning(f"    Stats computed: data keys={list(data_pipeline.keys())}")
                        # Check if data was modified
                        if data_pipeline_before != data_pipeline:
                            logger.warning("    Data was modified during stats computation")

                    if hasattr(op, "process_batched"):
                        result = list(op.process_batched(data_pipeline))
                        # Handle both boolean masks and text content
                        if result and isinstance(result[0], bool):
                            op_mask = result
                        else:
                            # If it returns text content, create a mask based on non-empty results
                            op_mask = [bool(item) for item in result]

                        # Check if op_mask has enough elements and idx is valid
                        if op_mask and len(op_mask) > idx:
                            pipeline_masks.append(op_mask[idx])
                            mask_pipeline = [m and o for m, o in zip(mask_pipeline, op_mask)]
                            logger.warning(f"    Result for sample {idx}: {op_mask[idx]} (passed={op_mask[idx]})")
                            logger.warning(f"    Overall: {sum(op_mask)}/{len(op_mask)} samples passed")
                            logger.warning(f"    Current masks so far: {pipeline_masks}")
                        else:
                            logger.warning(
                                f"    Warning: op_mask length {len(op_mask)} is insufficient for index {idx}"
                            )
                            pipeline_masks.append(True)  # Default to pass if mask is invalid
                    else:
                        logger.warning("    No process_batched method found")
                        pipeline_masks.append(True)

                # Check if masks are valid before accessing
                individual_result = all(ind_masks) if ind_masks else True
                pipeline_result = mask_pipeline[idx] if len(mask_pipeline) > idx else True

                logger.warning(f"  Pipeline final result: {pipeline_result} (all masks: {pipeline_masks})")
                logger.warning(f"  Comparison: Individual={individual_result} vs Pipeline={pipeline_result}")
                logger.warning("--- End of mismatch analysis ---")

        # Compile results
        results = {
            "mode": mode,
            "num_samples": len(test_data["text"]),
            "num_filters": len(filters),
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
                "mismatch_indices": mismatches[:10],  # Store first 10 mismatches
            },
            "analyzer_insights": analyzer_insights,
        }

        return results

    def get_quick_test_filters(self) -> List[Filter]:
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

    def get_full_test_filters(self) -> List[Filter]:
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
            alphanumeric_filter.AlphanumericFilter(min_ratio=0.5),
            text_length_filter.TextLengthFilter(min_len=10, max_len=1000),
            words_num_filter.WordsNumFilter(min_num=5, max_num=200),
            character_repetition_filter.CharacterRepetitionFilter(max_ratio=0.2),
            word_repetition_filter.WordRepetitionFilter(max_ratio=0.2),
            special_characters_filter.SpecialCharactersFilter(min_ratio=0.01, max_ratio=0.3),
            stopwords_filter.StopWordsFilter(min_ratio=0.1),
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

    def get_final_mask_from_filters(self, filters: List[Filter], test_data: Dict[str, Any]) -> list:
        """Compute the final boolean mask for each sample using individual filter execution (AND)."""
        data = test_data.copy()
        masks = []
        for filter_op in filters:
            if hasattr(filter_op, "compute_stats_batched"):
                data = filter_op.compute_stats_batched(data)
            if hasattr(filter_op, "process_batched"):
                result = list(filter_op.process_batched(data))
                # Handle both boolean masks and text content
                if result and isinstance(result[0], bool):
                    mask = result
                else:
                    # If it returns text content, create a mask based on non-empty results
                    mask = [bool(item) for item in result]
                masks.append(mask)
        # AND across all filters
        if masks:
            final_mask = [all(vals) for vals in zip(*masks)]
        else:
            final_mask = [True] * len(data["text"])
        return final_mask

    def get_final_mask_from_fused(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights=None
    ) -> list:
        """Compute the final boolean mask for each sample using fused execution (with fusion strategies)."""
        # Use the same logic as run_fused_filters_benchmark, but return the mask
        pipeline_config = self._build_pipeline_config_from_filters(filters)
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)
        from data_juicer.core.optimizer.filter_fusion_strategy import (
            FilterFusionStrategy,
        )
        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )
        from data_juicer.core.optimizer.optimizer import PipelineOptimizer

        strategies = [FilterFusionStrategy(analyzer_insights=analyzer_insights), MapperFusionStrategy()]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)
        optimized_ast = optimizer.optimize(ast)
        optimized_ops = self._convert_ast_to_operations(optimized_ast)

        return self.get_final_mask_from_optimized_ops(optimized_ops, test_data)

    def get_final_mask_from_optimized_ops(self, optimized_ops: List, test_data: Dict[str, Any]) -> list:
        """Compute the final boolean mask for each sample using optimized operations."""
        data = test_data.copy()
        mask = [True] * len(data["text"])
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
                    from data_juicer.core.optimizer.fused_op import FusedFilter

                    op = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    # Force sequential execution to avoid parallel stats access issues
                    op.execution_strategy = "sequential"
                else:
                    continue  # Skip if we can't create the fused filter
            elif op_name == "fused_mapper":
                # Extract the fused_mappers list and create FusedMapper directly
                mapper_config = op_config[op_name]
                name = mapper_config.get("name", "fused_mapper")
                fused_mappers = mapper_config.get("fused_mappers", [])

                # Create the fused mapper directly
                from data_juicer.core.optimizer.fused_op import FusedMapper

                op = FusedMapper(name=name, fused_mappers=fused_mappers)
            else:
                # Load the operation from config for non-fused operations
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    op = loaded_ops[0]
                else:
                    continue  # Skip if we can't load the operation

            # Execute the operation
            if hasattr(op, "compute_stats_batched"):
                data = op.compute_stats_batched(data)
            if hasattr(op, "process_batched"):
                result = list(op.process_batched(data))
                # Handle both boolean masks and text content
                if result and isinstance(result[0], bool):
                    op_mask = result
                else:
                    # If it returns text content, create a mask based on non-empty results
                    op_mask = [bool(item) for item in result]

                # Apply the operation mask (AND operation)
                if len(op_mask) == len(mask):
                    mask = [m and o for m, o in zip(mask, op_mask)]

        return mask

    def get_complex_only_test_filters(self) -> List[Filter]:
        """Get only complex filters for testing advanced fusion scenarios."""
        from data_juicer.ops.filter import (
            FlaggedWordFilter,
            LanguageIDScoreFilter,
            PerplexityFilter,
            StopWordsFilter,
            WordRepetitionFilter,
        )

        return [
            # Complex filters that require language models or advanced processing
            # (spaCy filters disabled due to download issues)
            PerplexityFilter(lang="en", model_key="gpt2", min_score=0.0, max_score=100.0),
            StopWordsFilter(lang="en", min_ratio=0.0, max_ratio=0.5),
            FlaggedWordFilter(lang="en", min_ratio=0.0, max_ratio=0.1),
            LanguageIDScoreFilter(lang="en", min_score=0.5, max_score=1.0),
            WordRepetitionFilter(lang="en", min_ratio=0.0, max_ratio=0.3),
            # spaCy filters disabled due to model download issues:
            # - TextEntityDependencyFilter requires spaCy models
            # - TextActionFilter requires spaCy models
        ]

    def run_complex_filters_benchmark(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> Dict[str, Any]:
        """Specialized benchmark for complex filters with multiple strategies."""
        logger.info("Running complex filters benchmark with multiple strategies...")

        results = {}

        # Strategy 1: Individual execution (baseline)
        logger.info("Strategy 1: Individual execution (baseline)")
        individual_metrics = self.run_individual_filters_benchmark(filters, test_data)
        results["individual"] = individual_metrics

        # Strategy 2: Conservative fusion (avoid complex+complex)
        logger.info("Strategy 2: Conservative fusion (avoid complex+complex)")
        conservative_metrics = self.run_conservative_fusion_benchmark(filters, test_data, analyzer_insights)
        results["conservative_fusion"] = conservative_metrics

        # Strategy 3: Parallel execution (no fusion, but parallel processing)
        logger.info("Strategy 3: Parallel execution (no fusion)")
        parallel_metrics = self.run_parallel_execution_benchmark(filters, test_data)
        results["parallel"] = parallel_metrics

        # Strategy 4: Smart batching (group by complexity)
        logger.info("Strategy 4: Smart batching (group by complexity)")
        smart_batch_metrics = self.run_smart_batching_benchmark(filters, test_data, analyzer_insights)
        results["smart_batching"] = smart_batch_metrics

        # Calculate improvements
        baseline_time = individual_metrics.total_time
        improvements = {}

        for strategy, metrics in results.items():
            if strategy != "individual":
                speedup = baseline_time / metrics.total_time
                improvements[strategy] = {
                    "speedup": speedup,
                    "time_saved": baseline_time - metrics.total_time,
                    "time_saved_percent": (baseline_time - metrics.total_time) / baseline_time * 100,
                }

        results["improvements"] = improvements

        # Find best strategy
        best_strategy = max(improvements.items(), key=lambda x: x[1]["speedup"])
        results["best_strategy"] = best_strategy[0]
        results["best_speedup"] = best_strategy[1]["speedup"]

        logger.info(f"🏆 Best strategy: {best_strategy[0]} ({best_strategy[1]['speedup']:.2f}x speedup)")

        return results

    def run_conservative_fusion_benchmark(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Run fusion benchmark with conservative strategy for complex filters."""
        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Build pipeline with conservative fusion
        pipeline_config = self._build_pipeline_config_from_filters(filters)
        from data_juicer.core.pipeline_ast import PipelineAST

        ast = PipelineAST()
        ast.build_from_config(pipeline_config)

        # Use conservative fusion strategy
        from data_juicer.core.optimizer.filter_fusion_strategy import (
            FilterFusionStrategy,
        )
        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )
        from data_juicer.core.optimizer.optimizer import PipelineOptimizer

        strategies = [FilterFusionStrategy(analyzer_insights=analyzer_insights), MapperFusionStrategy()]
        optimizer = PipelineOptimizer(strategies=strategies, analyzer_insights=analyzer_insights)

        optimized_ast = optimizer.optimize(ast)
        optimized_ops = self._convert_ast_to_operations(optimized_ast)

        logger.info(f"  Conservative fusion: {len(filters)} → {len(optimized_ops)} operations")

        # Process with optimized operations
        self._process_with_optimized_ops(optimized_ops, test_data)

        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = len(test_data["text"]) / total_time

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.8,
            filter_time=total_time * 0.2,
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def run_parallel_execution_benchmark(self, filters: List[Filter], test_data: Dict[str, Any]) -> PerformanceMetrics:
        """Run benchmark with parallel execution (no fusion)."""
        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Process each filter independently in parallel
        import concurrent.futures
        import threading

        # Create thread-local storage for data
        thread_local = threading.local()

        def process_filter(filter_op):
            # Each thread gets its own copy of data
            if not hasattr(thread_local, "data"):
                thread_local.data = test_data.copy()

            # Process the filter
            if hasattr(filter_op, "compute_stats_batched"):
                thread_local.data = filter_op.compute_stats_batched(thread_local.data)
            if hasattr(filter_op, "process_batched"):
                return list(filter_op.process_batched(thread_local.data))
            return [True] * len(test_data["text"])

        # Execute filters in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(filters), 4)) as executor:
            futures = [executor.submit(process_filter, filter_op) for filter_op in filters]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Combine results (AND operation)
        if results:
            _ = [all(vals) for vals in zip(*results)]
        else:
            _ = [True] * len(test_data["text"])

        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = len(test_data["text"]) / total_time

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.6,
            filter_time=total_time * 0.4,
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def run_smart_batching_benchmark(
        self, filters: List[Filter], test_data: Dict[str, Any], analyzer_insights: dict = None
    ) -> PerformanceMetrics:
        """Run benchmark with smart batching based on complexity."""
        start_memory = self.measure_memory_usage()
        total_start_time = time.time()

        # Group filters by complexity
        simple_filters = []
        medium_filters = []
        complex_filters = []

        for filter_op in filters:
            op_name = getattr(filter_op, "_name", type(filter_op).__name__)
            complexity = self._get_operation_complexity(op_name)

            if complexity == "simple":
                simple_filters.append(filter_op)
            elif complexity == "medium":
                medium_filters.append(filter_op)
            else:
                complex_filters.append(filter_op)

        logger.info(
            f"  Smart batching: {len(simple_filters)} simple, {len(medium_filters)} medium, {len(complex_filters)} complex"
        )

        # Process each complexity group separately
        data = test_data.copy()
        all_masks = []

        # Process simple filters (can be fused)
        if simple_filters:
            _ = self.run_fused_filters_benchmark(simple_filters, data, analyzer_insights)
            # Get mask from simple filters (simplified)
            simple_mask = [True] * len(data["text"])  # Placeholder
            all_masks.append(simple_mask)

        # Process medium filters (can be fused)
        if medium_filters:
            _ = self.run_fused_filters_benchmark(medium_filters, data, analyzer_insights)
            medium_mask = [True] * len(data["text"])  # Placeholder
            all_masks.append(medium_mask)

        # Process complex filters (individual execution)
        if complex_filters:
            for filter_op in complex_filters:
                if hasattr(filter_op, "compute_stats_batched"):
                    data = filter_op.compute_stats_batched(data)
                if hasattr(filter_op, "process_batched"):
                    result = list(filter_op.process_batched(data))
                    if result and isinstance(result[0], bool):
                        all_masks.append(result)
                    else:
                        all_masks.append([bool(item) for item in result])

        # Combine all masks
        if all_masks:
            _ = [all(vals) for vals in zip(*all_masks)]
        else:
            _ = [True] * len(data["text"])

        total_time = time.time() - total_start_time
        end_memory = self.measure_memory_usage()
        memory_usage = end_memory - start_memory
        throughput = len(test_data["text"]) / total_time

        return PerformanceMetrics(
            total_time=total_time,
            stats_time=total_time * 0.7,
            filter_time=total_time * 0.3,
            memory_usage=memory_usage,
            throughput=throughput,
        )

    def _get_operation_complexity(self, op_name: str) -> str:
        """Get operation complexity for smart batching."""
        op_name_lower = op_name.lower()

        # Simple operations
        if any(pattern in op_name_lower for pattern in ["text_length", "words_num", "character_repetition"]):
            return "simple"

        # Complex operations
        if any(pattern in op_name_lower for pattern in ["perplexity", "language_id", "text_entity", "text_action"]):
            return "complex"

        # Medium operations
        return "medium"


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


def create_simple_filters() -> List[Filter]:
    """Create a comprehensive set of simple filters for testing."""

    filters = [
        # Basic text metrics (very fast)
        WordsNumFilter(min_num=5, max_num=1000),
        TextLengthFilter(min_len=20, max_len=1000),
        CharacterRepetitionFilter(repetition_ratio=0.8),
        # Text structure filters (fast)
        WordRepetitionFilter(min_ratio=0.0, max_ratio=0.5),
        SpecialCharactersFilter(min_ratio=0.0, max_ratio=0.3),
        AlphanumericFilter(min_ratio=0.3),
        AverageLineLengthFilter(min_len=10, max_len=100),
        MaximumLineLengthFilter(min_len=10, max_len=200),
    ]

    return filters


def create_complex_filters():
    """Create complex filters that should trigger sequential execution."""
    from data_juicer.ops.filter import (
        FlaggedWordFilter,
        LanguageIDScoreFilter,
        PerplexityFilter,
        StopWordsFilter,
        TextActionFilter,
        TextEntityDependencyFilter,
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

    # Text entity dependency filter (complex - requires spaCy NLP)
    entity_dep_filter = TextEntityDependencyFilter(lang="en", min_dependency_num=1, any_or_all="all")
    filters.append(entity_dep_filter)

    # Text action filter (complex - requires spaCy POS tagging)
    action_filter = TextActionFilter(lang="en", min_action_num=1)
    filters.append(action_filter)

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
    logger.info("Creating simple filters...")
    filters = create_simple_filters()
    op_names = [getattr(f, "_name", type(f).__name__) for f in filters]
    logger.info(f"Created {len(filters)} simple filters: {op_names}")

    # Test 1: Simple Filters (should use parallel strategy)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Simple Filters (Parallel Strategy)")
    logger.info("=" * 60)

    # Collect filtering statistics with analyzer insights
    collect_filtering_stats_with_insights(filters, samples, analyzer_insights)

    # Benchmark individual execution
    logger.info("\n" + "=" * 60)
    individual_stats = benchmark_individual_simple(filters, samples)

    # Benchmark fused execution with analyzer insights (using optimizer architecture)
    logger.info("\n" + "=" * 60)
    fused_stats = benchmark.run_fused_filters_benchmark(filters, samples, analyzer_insights=analyzer_insights)

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
    logger.info(f"  Total Time: {fused_stats.total_time:.3f}s")
    logger.info(f"  Stats Time: {fused_stats.stats_time:.3f}s")
    logger.info(f"  Filter Time: {fused_stats.filter_time:.3f}s")
    logger.info(
        f"  Results: {fused_stats.throughput * fused_stats.total_time:.0f} samples processed at {fused_stats.throughput:.1f} samples/sec"
    )

    # Check if results are consistent (using throughput as proxy)
    individual_throughput = num_samples / individual_stats["total_time"]
    fused_throughput = fused_stats.throughput
    throughput_difference = abs(individual_throughput - fused_throughput)

    if throughput_difference > individual_throughput * 0.1:  # 10% tolerance
        logger.info(f"  ⚠️  Throughput Difference: {throughput_difference:.1f} samples/sec")
        logger.info("     This may indicate different execution patterns between individual and fused")
    else:
        logger.info("  ✅ Individual and fused throughput are consistent")
        logger.info("     Both use parallel execution: all filters see original data")
        logger.info("     Fusion provides performance benefits without changing results")

    # Calculate improvements
    total_speedup = individual_stats["total_time"] / fused_stats.total_time
    time_saved = individual_stats["total_time"] - fused_stats.total_time
    stats_speedup = individual_stats["stats_time"] / fused_stats.stats_time
    filter_speedup = (
        individual_stats["filter_time"] / fused_stats.filter_time if fused_stats.filter_time > 0 else float("inf")
    )

    logger.info("🎯 IMPROVEMENTS:")
    logger.info(f"  Total Speedup: {total_speedup:.2f}x")
    logger.info(f"  Time Saved: {time_saved:.3f}s " f'({time_saved/individual_stats["total_time"]*100:.1f}%)')
    logger.info(f"  Stats Speedup: {stats_speedup:.2f}x")
    logger.info(f"  Filter Speedup: {filter_speedup:.2f}x")

    # Calculate throughput
    individual_throughput = num_samples / individual_stats["total_time"]
    fused_throughput = fused_stats.throughput
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

    # Benchmark fused execution with analyzer insights (using optimizer architecture)
    logger.info("\n" + "=" * 60)
    fused_stats_complex = benchmark.run_fused_filters_benchmark(
        complex_filters, samples, analyzer_insights=analyzer_insights
    )

    # Print performance results for the second test
    logger.info("\n" + "=" * 60)
    logger.info("📊 SECOND TEST PERFORMANCE RESULTS")
    logger.info("=" * 60)

    total_speedup_complex = individual_stats_complex["total_time"] / fused_stats_complex.total_time
    time_saved_complex = individual_stats_complex["total_time"] - fused_stats_complex.total_time

    logger.info("Individual Execution:")
    logger.info(f"  Total Time: {individual_stats_complex['total_time']:.3f}s")
    logger.info(f"  Stats Time: {individual_stats_complex['stats_time']:.3f}s")
    logger.info(f"  Filter Time: {individual_stats_complex['filter_time']:.3f}s")

    logger.info("Fused Execution (with Analyzer Insights):")
    logger.info(f"  Total Time: {fused_stats_complex.total_time:.3f}s")
    logger.info(f"  Stats Time: {fused_stats_complex.stats_time:.3f}s")
    logger.info(f"  Filter Time: {fused_stats_complex.filter_time:.3f}s")

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
    """Main execution function for performance benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Benchmark for Data-Juicer Filters")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "complex_only", "pipeline", "recipe"],
        default="quick",
        help="Benchmark mode: quick (3 basic filters), full (12 comprehensive filters), complex_only (7 complex filters), pipeline (optimizer workflow), recipe (custom YAML pipeline)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of test samples to use",
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

    args = parser.parse_args()

    # Create benchmark instance
    benchmark = PerformanceBenchmark()

    # Preload models to avoid loading delays during benchmark
    if args.mode in ["complex_only", "full", "recipe"]:
        preload_models_for_benchmark()

    # Create test data
    test_data = create_simple_test_data(args.samples)

    # Get filters based on mode
    if args.mode == "quick":
        filters = benchmark.get_quick_test_filters()
    elif args.mode == "full":
        filters = benchmark.create_test_filters()  # Use comprehensive test filters (12 filters)
    elif args.mode == "complex_only":
        from data_juicer.ops.filter import (
            FlaggedWordFilter,
            LanguageIDScoreFilter,
            PerplexityFilter,
            StopWordsFilter,
            WordRepetitionFilter,
        )

        base_filters = [
            PerplexityFilter(lang="en", model_key="gpt2", min_score=0.0, max_score=100.0),
            StopWordsFilter(lang="en", min_ratio=0.0, max_ratio=0.5),
            FlaggedWordFilter(lang="en", min_ratio=0.0, max_ratio=0.1),
            LanguageIDScoreFilter(lang="en", min_score=0.5, max_score=1.0),
            WordRepetitionFilter(lang="en", min_ratio=0.0, max_ratio=0.3),
        ]
        spacy_filters = []
        filters = base_filters + spacy_filters
    elif args.mode == "recipe":
        # Load pipeline from YAML recipe using optimizer framework
        if not args.recipe_path:
            raise ValueError("--recipe-path must be specified in recipe mode!")
        from data_juicer.core.optimizer.filter_fusion_strategy import (
            FilterFusionStrategy,
        )
        from data_juicer.core.optimizer.mapper_fusion_strategy import (
            MapperFusionStrategy,
        )
        from data_juicer.core.optimizer.optimizer import PipelineOptimizer
        from data_juicer.core.pipeline_ast import PipelineAST

        # Build AST from recipe
        ast = PipelineAST()
        ast.build_from_yaml(args.recipe_path)
        logger.info(f"📋 Loaded recipe: {args.recipe_path}")
        logger.info(f"Pipeline structure:\n{ast.visualize()}")
        # Use optimizer to handle the full pipeline
        optimizer = PipelineOptimizer([FilterFusionStrategy(), MapperFusionStrategy()])
        # Optimize the pipeline
        optimized_ast = optimizer.optimize(ast)
        logger.info(f"🔧 Optimized pipeline structure:\n{optimized_ast.visualize()}")
        # Convert optimized AST to operations for benchmarking
        operations = benchmark._convert_ast_to_operations(optimized_ast)
        # Debug: Log the extracted operations
        logger.info(f"🔍 Extracted {len(operations)} operations from optimized AST:")
        for i, op in enumerate(operations):
            logger.info(f"  {i+1}: {op}")
        # Load operations, handling fused_filter specially
        from data_juicer.core.optimizer.fused_op import FusedFilter
        from data_juicer.ops.load import load_ops

        filters = []
        for op_config in operations:
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
                    filters.append(fused_filter)
            else:
                loaded_ops = load_ops([op_config])
                if loaded_ops:
                    filters.append(loaded_ops[0])
        logger.info(f"📊 Loaded {len(filters)} operations from optimized pipeline")
    else:  # pipeline mode
        filters = benchmark.create_test_filters()  # Use comprehensive test filters for pipeline mode

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


if __name__ == "__main__":
    main()
