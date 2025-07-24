"""
Partition Size Optimizer for DataJuicer

This module automatically configures optimal partition sizes based on:
1. Data modality (text, image, audio, video, multimodal)
2. Dataset characteristics (file sizes, complexity)
3. Available system resources
4. Processing pipeline complexity
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from loguru import logger


class ModalityType(Enum):
    """Supported data modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""

    modality: ModalityType
    default_partition_size: int
    max_partition_size: int
    max_partition_size_mb: int
    memory_multiplier: float  # Memory usage multiplier compared to text
    complexity_multiplier: float  # Processing complexity multiplier
    description: str


class PartitionSizeOptimizer:
    """Automatically optimizes partition sizes based on data characteristics."""

    # Default configurations for different modalities
    MODALITY_CONFIGS = {
        ModalityType.TEXT: ModalityConfig(
            modality=ModalityType.TEXT,
            default_partition_size=200,
            max_partition_size=1000,
            max_partition_size_mb=32,
            memory_multiplier=1.0,
            complexity_multiplier=1.0,
            description="Text data - efficient processing, low memory usage",
        ),
        ModalityType.IMAGE: ModalityConfig(
            modality=ModalityType.IMAGE,
            default_partition_size=50,
            max_partition_size=200,
            max_partition_size_mb=128,
            memory_multiplier=5.0,
            complexity_multiplier=3.0,
            description="Image data - moderate memory usage, image processing overhead",
        ),
        ModalityType.AUDIO: ModalityConfig(
            modality=ModalityType.AUDIO,
            default_partition_size=30,
            max_partition_size=100,
            max_partition_size_mb=256,
            memory_multiplier=8.0,
            complexity_multiplier=5.0,
            description="Audio data - high memory usage, audio processing overhead",
        ),
        ModalityType.VIDEO: ModalityConfig(
            modality=ModalityType.VIDEO,
            default_partition_size=10,
            max_partition_size=50,
            max_partition_size_mb=512,
            memory_multiplier=20.0,
            complexity_multiplier=15.0,
            description="Video data - very high memory usage, complex processing",
        ),
        ModalityType.MULTIMODAL: ModalityConfig(
            modality=ModalityType.MULTIMODAL,
            default_partition_size=20,
            max_partition_size=100,
            max_partition_size_mb=256,
            memory_multiplier=10.0,
            complexity_multiplier=8.0,
            description="Multimodal data - combination of multiple modalities",
        ),
    }

    def __init__(self, cfg):
        """Initialize the optimizer with configuration."""
        self.cfg = cfg
        self.text_key = getattr(cfg, "text_key", "text")
        self.image_key = getattr(cfg, "image_key", "images")
        self.audio_key = getattr(cfg, "audio_key", "audios")
        self.video_key = getattr(cfg, "video_key", "videos")

    def detect_modality(self, sample: Dict) -> ModalityType:
        """Detect the primary modality of a sample."""
        modalities = []

        # Check for text
        if self.text_key in sample and sample[self.text_key]:
            modalities.append(ModalityType.TEXT)

        # Check for images
        if self.image_key in sample and sample[self.image_key]:
            modalities.append(ModalityType.IMAGE)

        # Check for audio
        if self.audio_key in sample and sample[self.audio_key]:
            modalities.append(ModalityType.AUDIO)

        # Check for video
        if self.video_key in sample and sample[self.video_key]:
            modalities.append(ModalityType.VIDEO)

        # Determine primary modality
        if len(modalities) > 1:
            return ModalityType.MULTIMODAL
        elif len(modalities) == 1:
            return modalities[0]
        else:
            # Default to text if no modality detected
            return ModalityType.TEXT

    def analyze_dataset_characteristics(self, dataset) -> Dict:
        """Analyze dataset characteristics to inform partition sizing."""
        logger.info("Analyzing dataset characteristics for partition optimization...")

        # For Ray Datasets, we need to handle them differently
        try:
            # Try to get dataset size
            if hasattr(dataset, "count"):
                total_samples = dataset.count()
            elif hasattr(dataset, "__len__"):
                total_samples = len(dataset)
            else:
                # Try to estimate from available methods
                try:
                    # For Ray Datasets, try to get a sample and estimate
                    if hasattr(dataset, "take"):
                        sample_batch = dataset.take(1)
                        if sample_batch:
                            # Estimate based on available info or use conservative default
                            total_samples = 1000  # Conservative estimate
                            logger.info("Using conservative dataset size estimate of 1000 samples")
                        else:
                            total_samples = 100
                            logger.info("Empty dataset detected, using 100 samples")
                    else:
                        total_samples = 1000
                        logger.warning("Could not determine dataset size, using conservative estimate of 1000 samples")
                except Exception:
                    total_samples = 1000
                    logger.warning("Could not determine dataset size, using conservative estimate of 1000 samples")
        except Exception as e:
            logger.warning(f"Could not determine dataset size: {e}, using conservative estimate of 1000 samples")
            total_samples = 1000

        # Sample a subset for analysis
        sample_size = min(100, total_samples)

        try:
            # For Ray Datasets, use take() to get samples
            if hasattr(dataset, "take"):
                samples = dataset.take(sample_size)
                logger.info(f"Successfully sampled {len(samples)} samples from Ray Dataset")
            else:
                # Fallback for other dataset types
                samples = dataset.select(range(sample_size))
                logger.info(f"Successfully sampled {len(samples)} samples from dataset")
        except Exception as e:
            logger.warning(f"Could not sample dataset: {e}, using default analysis")
            # Return default characteristics
            return {
                "primary_modality": ModalityType.TEXT,
                "modality_distribution": {ModalityType.TEXT: 1},
                "avg_text_length": 500,
                "avg_images_per_sample": 0,
                "avg_audio_per_sample": 0,
                "avg_video_per_sample": 0,
                "total_samples": total_samples,
                "sample_size_analyzed": 0,
            }

        modality_counts = {modality: 0 for modality in ModalityType}
        total_text_length = 0
        total_image_count = 0
        total_audio_count = 0
        total_video_count = 0

        for sample in samples:
            modality = self.detect_modality(sample)
            modality_counts[modality] += 1

            # Analyze text characteristics
            if self.text_key in sample and sample[self.text_key]:
                if isinstance(sample[self.text_key], str):
                    total_text_length += len(sample[self.text_key])
                elif isinstance(sample[self.text_key], list):
                    total_text_length += sum(len(t) for t in sample[self.text_key])

            # Count media files
            if self.image_key in sample and sample[self.image_key]:
                total_image_count += len(sample[self.image_key])
            if self.audio_key in sample and sample[self.audio_key]:
                total_audio_count += len(sample[self.audio_key])
            if self.video_key in sample and sample[self.video_key]:
                total_video_count += len(sample[self.video_key])

        # Calculate averages
        avg_text_length = total_text_length / sample_size if sample_size > 0 else 0
        avg_images_per_sample = total_image_count / sample_size if sample_size > 0 else 0
        avg_audio_per_sample = total_audio_count / sample_size if sample_size > 0 else 0
        avg_video_per_sample = total_video_count / sample_size if sample_size > 0 else 0

        # Determine primary modality
        primary_modality = max(modality_counts.items(), key=lambda x: x[1])[0]

        characteristics = {
            "primary_modality": primary_modality,
            "modality_distribution": modality_counts,
            "avg_text_length": avg_text_length,
            "avg_images_per_sample": avg_images_per_sample,
            "avg_audio_per_sample": avg_audio_per_sample,
            "avg_video_per_sample": avg_video_per_sample,
            "total_samples": len(dataset),
            "sample_size_analyzed": sample_size,
        }

        logger.info(f"Dataset analysis complete:")
        logger.info(f"  Primary modality: {primary_modality.value}")
        logger.info(f"  Modality distribution: {modality_counts}")
        logger.info(f"  Avg text length: {avg_text_length:.0f} chars")
        logger.info(f"  Avg images per sample: {avg_images_per_sample:.1f}")
        logger.info(f"  Avg audio per sample: {avg_audio_per_sample:.1f}")
        logger.info(f"  Avg video per sample: {avg_video_per_sample:.1f}")

        return characteristics

    def analyze_processing_complexity(self, process_pipeline: List) -> float:
        """Analyze the complexity of the processing pipeline."""
        complexity_score = 1.0

        # Count operations by type
        op_counts = {}
        for op in process_pipeline:
            if isinstance(op, dict):
                op_name = list(op.keys())[0]
                op_counts[op_name] = op_counts.get(op_name, 0) + 1

        # Adjust complexity based on operation types
        for op_name, count in op_counts.items():
            # High complexity operations
            if any(
                keyword in op_name.lower()
                for keyword in ["embedding", "similarity", "model", "neural", "vision", "audio"]
            ):
                complexity_score *= 1.2**count
            # Medium complexity operations
            elif any(keyword in op_name.lower() for keyword in ["filter", "deduplicator", "mapper"]):
                complexity_score *= 1.1**count
            # Low complexity operations (text cleaning, etc.)
            else:
                complexity_score *= 1.05**count

        logger.info(f"Processing complexity score: {complexity_score:.2f}")
        return complexity_score

    def get_optimal_partition_size(self, dataset, process_pipeline: List) -> Tuple[int, int]:
        """Get optimal partition size and max size based on dataset characteristics."""

        # Analyze dataset
        characteristics = self.analyze_dataset_characteristics(dataset)
        primary_modality = characteristics["primary_modality"]

        # Get base configuration for the modality
        base_config = self.MODALITY_CONFIGS[primary_modality]

        # Analyze processing complexity
        complexity_multiplier = self.analyze_processing_complexity(process_pipeline)

        # Calculate optimal partition size based on modality
        if primary_modality == ModalityType.TEXT:
            # Use intelligent text partition sizing
            optimal_size = self.calculate_text_partition_size(
                characteristics["avg_text_length"], characteristics["total_samples"], complexity_multiplier
            )
        else:
            # Use standard calculation for other modalities
            optimal_size = int(base_config.default_partition_size / complexity_multiplier)
            optimal_size = max(10, min(optimal_size, base_config.max_partition_size))

        # Calculate optimal max size in MB
        optimal_max_size_mb = int(base_config.max_partition_size_mb / complexity_multiplier)
        optimal_max_size_mb = max(16, min(optimal_max_size_mb, 1024))

        # Apply additional adjustments based on dataset size
        if characteristics["total_samples"] < 1000:
            # Small dataset - use smaller partitions for better granularity
            optimal_size = max(10, optimal_size // 2)
        elif characteristics["total_samples"] > 100000:
            # Large dataset - can use larger partitions
            optimal_size = min(base_config.max_partition_size, optimal_size * 2)

        # Apply adjustments based on text length
        if characteristics["avg_text_length"] > 10000:
            # Long text - reduce partition size
            optimal_size = max(10, optimal_size // 2)
        elif characteristics["avg_text_length"] < 100:
            # Short text - can use larger partitions
            optimal_size = min(base_config.max_partition_size, optimal_size * 2)

        logger.info(f"Optimal partition configuration:")
        logger.info(f"  Size: {optimal_size} samples")
        logger.info(f"  Max size: {optimal_max_size_mb} MB")
        logger.info(f"  Based on: {primary_modality.value} modality")
        logger.info(f"  Complexity multiplier: {complexity_multiplier:.2f}")

        return optimal_size, optimal_max_size_mb

    def get_partition_recommendations(self, dataset, process_pipeline: List) -> Dict:
        """Get comprehensive partition recommendations."""
        optimal_size, optimal_max_size_mb = self.get_optimal_partition_size(dataset, process_pipeline)
        characteristics = self.analyze_dataset_characteristics(dataset)

        recommendations = {
            "recommended_partition_size": optimal_size,
            "recommended_max_size_mb": optimal_max_size_mb,
            "primary_modality": characteristics["primary_modality"].value,
            "reasoning": {
                "modality": f"Based on {characteristics['primary_modality'].value} modality",
                "complexity": f"Processing complexity factor: {self.analyze_processing_complexity(process_pipeline):.2f}",
                "dataset_size": f"Dataset size: {characteristics['total_samples']} samples",
                "text_length": f"Average text length: {characteristics['avg_text_length']:.0f} characters",
            },
            "modality_configs": {
                modality.value: {
                    "default_size": config.default_partition_size,
                    "max_size": config.max_partition_size,
                    "max_size_mb": config.max_partition_size_mb,
                    "description": config.description,
                }
                for modality, config in self.MODALITY_CONFIGS.items()
            },
        }

        return recommendations

    def calculate_text_partition_size(self, avg_text_length: float, total_samples: int, complexity_score: float) -> int:
        """
        Calculate optimal text partition size based on actual data characteristics.

        Factors considered:
        1. Text length (longer text = smaller partitions)
        2. Dataset size (larger datasets can use larger partitions)
        3. Processing complexity (complex operations = smaller partitions)
        4. Memory constraints (estimated memory usage)
        """
        # Base partition size for text
        base_size = 200

        # Adjust for text length
        if avg_text_length > 10000:
            # Very long text (articles, documents) - use smaller partitions
            length_factor = 0.3
        elif avg_text_length > 5000:
            # Long text (paragraphs) - moderate reduction
            length_factor = 0.6
        elif avg_text_length > 1000:
            # Medium text (sentences) - slight reduction
            length_factor = 0.8
        elif avg_text_length < 100:
            # Very short text (tweets, labels) - can use larger partitions
            length_factor = 1.5
        else:
            # Normal text length
            length_factor = 1.0

        # Adjust for dataset size
        if total_samples > 1000000:
            # Very large dataset - can use larger partitions
            size_factor = 1.5
        elif total_samples > 100000:
            # Large dataset - moderate increase
            size_factor = 1.2
        elif total_samples < 1000:
            # Small dataset - use smaller partitions for better granularity
            size_factor = 0.7
        else:
            # Medium dataset
            size_factor = 1.0

        # Adjust for processing complexity
        complexity_factor = 1.0 / complexity_score

        # Calculate optimal size
        optimal_size = int(base_size * length_factor * size_factor * complexity_factor)

        # Apply bounds
        min_size = 10
        max_size = 1000

        optimal_size = max(min_size, min(optimal_size, max_size))

        logger.info(f"Text partition size calculation:")
        logger.info(f"  Base size: {base_size}")
        logger.info(f"  Avg text length: {avg_text_length:.0f} chars (factor: {length_factor:.2f})")
        logger.info(f"  Dataset size: {total_samples} samples (factor: {size_factor:.2f})")
        logger.info(f"  Complexity score: {complexity_score:.2f} (factor: {complexity_factor:.2f})")
        logger.info(f"  Optimal size: {optimal_size} samples")

        return optimal_size


def auto_configure_partition_size(cfg, dataset, process_pipeline: List) -> Dict:
    """
    Automatically configure partition size based on dataset characteristics.

    Args:
        cfg: Configuration object
        dataset: Dataset to analyze
        process_pipeline: List of processing operations

    Returns:
        Dict with recommended partition configuration
    """
    optimizer = PartitionSizeOptimizer(cfg)
    recommendations = optimizer.get_partition_recommendations(dataset, process_pipeline)

    # Update configuration with recommendations
    if not hasattr(cfg, "partition"):
        cfg.partition = {}

    cfg.partition["size"] = recommendations["recommended_partition_size"]
    cfg.partition["max_size_mb"] = recommendations["recommended_max_size_mb"]

    logger.info("Auto-configured partition settings:")
    logger.info(f"  partition.size: {cfg.partition['size']}")
    logger.info(f"  partition.max_size_mb: {cfg.partition['max_size_mb']}")

    return recommendations
