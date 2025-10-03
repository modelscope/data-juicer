#!/usr/bin/env python3
"""
Comprehensive workload suite for benchmarking different scenarios.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class WorkloadDefinition:
    """Definition of a benchmark workload."""

    name: str
    description: str
    dataset_path: str
    config_path: str
    expected_samples: int
    modality: str  # text, image, video, audio, multimodal
    complexity: str  # simple, medium, complex
    estimated_duration_minutes: int
    resource_requirements: Dict[str, Any]

    def __post_init__(self):
        """Validate workload definition."""
        if not Path(self.dataset_path).exists():
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
        if not Path(self.config_path).exists():
            logger.warning(f"Config path does not exist: {self.config_path}")


class WorkloadSuite:
    """Comprehensive suite of benchmark workloads."""

    def __init__(self):
        self.workloads = {}
        self._initialize_workloads()

    def _initialize_workloads(self):
        """Initialize all available workloads."""

        # Text workloads
        self.workloads["text_simple"] = WorkloadDefinition(
            name="text_simple",
            description="Simple text processing with basic filters",
            dataset_path="demos/data/text_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=1000,
            modality="text",
            complexity="simple",
            estimated_duration_minutes=2,
            resource_requirements={"memory_gb": 2, "cpu_cores": 2},
        )

        self.workloads["text_complex"] = WorkloadDefinition(
            name="text_complex",
            description="Complex text processing with multiple operations",
            dataset_path="demos/data/text_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=1000,
            modality="text",
            complexity="complex",
            estimated_duration_minutes=10,
            resource_requirements={"memory_gb": 8, "cpu_cores": 8},
        )

        # Image workloads
        self.workloads["image_simple"] = WorkloadDefinition(
            name="image_simple",
            description="Simple image processing",
            dataset_path="demos/data/image_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=500,
            modality="image",
            complexity="simple",
            estimated_duration_minutes=5,
            resource_requirements={"memory_gb": 4, "cpu_cores": 4, "gpu": True},
        )

        self.workloads["image_complex"] = WorkloadDefinition(
            name="image_complex",
            description="Complex image processing with ML models",
            dataset_path="demos/data/image_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=500,
            modality="image",
            complexity="complex",
            estimated_duration_minutes=20,
            resource_requirements={"memory_gb": 16, "cpu_cores": 8, "gpu": True},
        )

        # Video workloads
        self.workloads["video_simple"] = WorkloadDefinition(
            name="video_simple",
            description="Simple video processing",
            dataset_path="demos/data/video_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=100,
            modality="video",
            complexity="simple",
            estimated_duration_minutes=15,
            resource_requirements={"memory_gb": 8, "cpu_cores": 8, "gpu": True},
        )

        self.workloads["video_complex"] = WorkloadDefinition(
            name="video_complex",
            description="Complex video processing with frame analysis",
            dataset_path="demos/data/video_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=100,
            modality="video",
            complexity="complex",
            estimated_duration_minutes=45,
            resource_requirements={"memory_gb": 32, "cpu_cores": 16, "gpu": True},
        )

        # Audio workloads
        self.workloads["audio_simple"] = WorkloadDefinition(
            name="audio_simple",
            description="Simple audio processing",
            dataset_path="demos/data/audio_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=200,
            modality="audio",
            complexity="simple",
            estimated_duration_minutes=8,
            resource_requirements={"memory_gb": 4, "cpu_cores": 4},
        )

        # Multimodal workloads
        self.workloads["multimodal_medium"] = WorkloadDefinition(
            name="multimodal_medium",
            description="Medium complexity multimodal processing",
            dataset_path="demos/data/multimodal_data.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=300,
            modality="multimodal",
            complexity="medium",
            estimated_duration_minutes=25,
            resource_requirements={"memory_gb": 16, "cpu_cores": 8, "gpu": True},
        )

        # Performance stress tests
        self.workloads["stress_test"] = WorkloadDefinition(
            name="stress_test",
            description="High-volume stress test",
            dataset_path="demos/data/large_dataset.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=10000,
            modality="text",
            complexity="complex",
            estimated_duration_minutes=60,
            resource_requirements={"memory_gb": 32, "cpu_cores": 16},
        )

    def get_workload(self, name: str) -> Optional[WorkloadDefinition]:
        """Get a specific workload by name."""
        return self.workloads.get(name)

    def get_workloads_by_modality(self, modality: str) -> List[WorkloadDefinition]:
        """Get all workloads for a specific modality."""
        return [w for w in self.workloads.values() if w.modality == modality]

    def get_workloads_by_complexity(self, complexity: str) -> List[WorkloadDefinition]:
        """Get all workloads for a specific complexity level."""
        return [w for w in self.workloads.values() if w.complexity == complexity]

    def get_all_workloads(self) -> List[WorkloadDefinition]:
        """Get all available workloads."""
        return list(self.workloads.values())

    def get_workload_names(self) -> List[str]:
        """Get names of all available workloads."""
        return list(self.workloads.keys())

    def validate_workloads(self) -> Dict[str, List[str]]:
        """Validate all workloads and return any issues."""
        issues = {}

        for name, workload in self.workloads.items():
            workload_issues = []

            if not Path(workload.dataset_path).exists():
                workload_issues.append(f"Dataset not found: {workload.dataset_path}")

            if not Path(workload.config_path).exists():
                workload_issues.append(f"Config not found: {workload.config_path}")

            if workload.expected_samples <= 0:
                workload_issues.append("Expected samples must be positive")

            if workload.estimated_duration_minutes <= 0:
                workload_issues.append("Estimated duration must be positive")

            if workload_issues:
                issues[name] = workload_issues

        return issues


# Global workload suite instance
WORKLOAD_SUITE = WorkloadSuite()
