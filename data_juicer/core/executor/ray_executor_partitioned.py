"""
Enhanced Partitioned Ray Executor for Fault-Tolerant Large Dataset Processing

This module implements a comprehensive partitioned execution strategy for Ray mode that:
1. Splits large datasets into manageable partitions
2. Processes each partition independently with fault tolerance
3. Provides comprehensive checkpointing and recovery mechanisms
4. Enables partial failure recovery without losing all progress
5. Preserves mapping between original dataset and partitions
6. Provides comprehensive event logging and monitoring
7. Supports multiple storage formats (Parquet, Arrow, JSONL)
8. Offers real-time status monitoring and debugging
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.adapter import Adapter
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


def _process_single_batch(
    file_path: str, ops: List, partition_id: int, work_dir: str, cfg: Namespace
) -> Dict[str, Any]:
    """
    Process a single batch as a Ray dataset.

    Args:
        file_path: Path to the input file
        ops: List of operations to apply
        partition_id: ID of the partition
        work_dir: Working directory
        cfg: Configuration object

    Returns:
        Dictionary containing processing results
    """
    try:
        logger.info(f"Processing file {partition_id}: {file_path}")

        # Create results directory for this partition
        results_dir = os.path.join(work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create DatasetBuilder instance for this partition
        partition_cfg = Namespace(**vars(cfg))
        partition_cfg.dataset_path = file_path  # Override dataset_path for this partition
        datasetbuilder = DatasetBuilder(partition_cfg, executor_type="ray")

        # Load dataset using DatasetBuilder for proper handling
        dataset = datasetbuilder.load_dataset()

        # Apply all operations
        processed_dataset = dataset.process(ops)

        # Get sample count (avoid calling .count() during convergence to prevent memory issues)
        sample_count = 0  # Will be calculated later if needed

        logger.info(f"File {partition_id} processed successfully")

        return {
            "partition_id": partition_id,
            "input_path": file_path,
            "dataset": processed_dataset,  # Return dataset object instead of file path
            "success": True,
            "sample_count": sample_count,
            "error": None,
        }

    except Exception as e:
        logger.error(f"File {partition_id} processing failed: {e}")
        return {
            "partition_id": partition_id,
            "input_path": file_path,
            "output_path": None,
            "success": False,
            "sample_count": 0,
            "error": str(e),
        }


class CheckpointStrategy(Enum):
    """Checkpoint strategies for controlling when to create checkpoints."""

    EVERY_OP = "every_op"  # Checkpoint after every operation
    EVERY_PARTITION = "every_partition"  # Checkpoint only at partition completion
    EVERY_N_OPS = "every_n_ops"  # Checkpoint after every N operations
    MANUAL = "manual"  # Checkpoint only after specified operations
    DISABLED = "disabled"  # Disable checkpointing entirely


@dataclass
class PartitionMetadata:
    """Metadata for tracking partition information."""

    partition_id: int
    original_start_idx: int
    original_end_idx: int
    sample_count: int
    file_size_bytes: int
    checksum: str
    created_timestamp: float
    file_path: Optional[str] = None  # Path to the original file for file-based partitioning
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class ProcessingEvent:
    """Event log entry for processing operations."""

    event_id: str
    event_type: str  # partition_start, partition_complete, operation_checkpoint, error, etc.
    timestamp: float
    partition_id: Optional[int] = None
    operation_name: Optional[str] = None
    operation_idx: Optional[int] = None
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


@dataclass
class DatasetMapping:
    """Mapping information between original dataset and partitions."""

    original_dataset_path: str
    original_dataset_size: int
    partition_count: int
    partition_size: int
    mapping_version: str = "1.0"
    created_timestamp: Optional[float] = None
    partitions: Optional[List[PartitionMetadata]] = None

    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = time.time()
        if self.partitions is None:
            self.partitions = []


class PartitionedRayExecutor(ExecutorBase, EventLoggingMixin, DAGExecutionMixin):
    """
    Fault-tolerant Ray executor with partitioning optimization.

    Features:
    - Automatic dataset partitioning for fault tolerance
    - Independent partition processing with recovery
    - Checkpointing at partition level
    - Partial failure recovery
    - Progress tracking and resumption
    - Preserved mapping between original dataset and partitions
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """Initialize the partitioned Ray executor."""
        super().__init__(cfg)

        self.executor_type = "ray_partitioned"
        self.work_dir = self.cfg.work_dir
        self.adapter = Adapter(self.cfg)

        # Initialize EventLoggingMixin for job management and event logging
        # Do this after work_dir is set
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)

        # Override strategy methods for partitioned execution
        self._override_strategy_methods()

        # Simplified configuration for natural file partitioning
        # With natural files, we don't need partition size parameters
        logger.info("Using natural file partitioning - partition boundaries determined by input files")

        # Retry configuration (fixed defaults)
        self.max_retries = 3
        self.retry_backoff = "exponential"

        # Intermediate storage configuration (includes file lifecycle management)
        intermediate_storage_config = getattr(self.cfg, "intermediate_storage", {})
        self.storage_format = intermediate_storage_config.get(
            "format", "parquet"
        )  # parquet, arrow, jsonl - for disk storage
        self.storage_compression = intermediate_storage_config.get("compression", "snappy")

        # File lifecycle management (now part of intermediate_storage config)
        self.preserve_intermediate_data = intermediate_storage_config.get("preserve_intermediate_data") or getattr(
            self.cfg, "preserve_intermediate_data", False
        )
        self.cleanup_temp_files = intermediate_storage_config.get("cleanup_temp_files", True)
        self.cleanup_on_success = intermediate_storage_config.get("cleanup_on_success", False)
        self.retention_policy = intermediate_storage_config.get("retention_policy", "keep_all")
        self.max_retention_days = intermediate_storage_config.get("max_retention_days", 7)

        # Partition writing configuration
        self.write_partitions = intermediate_storage_config.get("write_partitions", True)
        logger.info(f"Partition writing: {'enabled' if self.write_partitions else 'disabled'}")

        # Checkpoint configuration
        checkpoint_cfg = getattr(self.cfg, "checkpoint", None)
        if checkpoint_cfg:
            self.checkpoint_enabled = getattr(checkpoint_cfg, "enabled", True)

            # Parse checkpoint strategy with validation
            strategy_str = getattr(checkpoint_cfg, "strategy", "every_op")
            try:
                self.checkpoint_strategy = CheckpointStrategy(strategy_str)
            except ValueError:
                logger.warning(f"Unknown checkpoint strategy: {strategy_str}, defaulting to EVERY_OP")
                self.checkpoint_strategy = CheckpointStrategy.EVERY_OP

            self.checkpoint_n_ops = getattr(checkpoint_cfg, "n_ops", 1)
            self.checkpoint_op_names = getattr(checkpoint_cfg, "op_names", [])
        else:
            self.checkpoint_enabled = False
            self.checkpoint_strategy = CheckpointStrategy.DISABLED
            self.checkpoint_n_ops = 1
            self.checkpoint_op_names = []

        # If strategy is DISABLED, disable checkpointing regardless of enabled flag
        if self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            self.checkpoint_enabled = False

        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # DatasetBuilder will be created per partition for proper isolation

        # Initialize RayExporter for final output
        logger.info("Preparing exporter...")
        self.exporter = RayExporter(
            self.cfg.export_path,
            keep_stats_in_res_ds=getattr(self.cfg, "keep_stats_in_res_ds", True),
            keep_hashes_in_res_ds=getattr(self.cfg, "keep_hashes_in_res_ds", False),
        )

        # Use resolved directory paths from config (already handled by config.py)
        self.partitions_dir = self.cfg.partition_dir
        self.checkpoint_dir = self.cfg.checkpoint_dir
        self.results_dir = self.cfg.results_dir
        self.metadata_dir = self.cfg.metadata_dir
        self.logs_dir = self.cfg.event_log_dir
        self.events_file = self.cfg.event_log_file
        self.summary_file = os.path.join(self.logs_dir, "processing_summary.json")

        # Create directories (already created by config.py, but ensure they exist)
        for dir_path in [
            self.partitions_dir,
            self.checkpoint_dir,
            self.results_dir,
            self.metadata_dir,
            self.logs_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize event logging summary
        self.event_summary = {
            "start_time": time.time(),
            "total_partitions": 0,
            "completed_partitions": 0,
            "failed_partitions": 0,
            "total_operations": 0,
            "completed_operations": 0,
            "failed_operations": 0,
            "checkpoints_created": 0,
            "total_processing_time": 0,
            "errors": [],
        }

        # Initialize processing events tracking
        self.processing_events = []
        self.event_lock = threading.Lock()

        # Dataset mapping
        self.dataset_mapping: Optional[DatasetMapping] = None

    def _should_checkpoint(self, op_idx: int, op_name: str, partition_id: int) -> bool:
        """Determine if checkpoint should be created based on configuration strategy."""
        if not self.checkpoint_enabled:
            return False

        if self.checkpoint_strategy == CheckpointStrategy.EVERY_OP:
            return True
        elif self.checkpoint_strategy == CheckpointStrategy.EVERY_PARTITION:
            return False  # Will be handled at partition completion
        elif self.checkpoint_strategy == CheckpointStrategy.EVERY_N_OPS:
            return (op_idx + 1) % self.checkpoint_n_ops == 0
        elif self.checkpoint_strategy == CheckpointStrategy.MANUAL:
            return op_name in self.checkpoint_op_names
        elif self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            return False
        else:
            logger.warning(f"Unknown checkpoint strategy: {self.checkpoint_strategy}, defaulting to every_op")
            return True

    def _log_processing_event(self, event: ProcessingEvent):
        """Log a processing event."""
        with self.event_lock:
            self.processing_events.append(event)
            # Also log to file if available
            if hasattr(self, "events_file"):
                with open(self.events_file, "a") as f:
                    f.write(json.dumps(event.__dict__) + "\n")

    def _finalize_event_summary(self):
        """Finalize and save the processing summary."""
        self.event_summary["end_time"] = time.time()
        self.event_summary["total_processing_time"] = self.event_summary["end_time"] - self.event_summary["start_time"]

        with open(self.summary_file, "w") as f:
            json.dump(self.event_summary, f, indent=2)

    def get_events(self, event_type: Optional[str] = None, partition_id: Optional[int] = None) -> List[ProcessingEvent]:
        """Retrieve events with optional filtering."""
        events = []
        if os.path.exists(self.events_file):
            with open(self.events_file, "r") as f:
                for line in f:
                    event_data = json.loads(line.strip())
                    event = ProcessingEvent(**event_data)

                    if event_type and event.event_type != event_type:
                        continue
                    if partition_id is not None and event.partition_id != partition_id:
                        continue

                    events.append(event)
        return events

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of the current processing status."""
        completed_partitions = (
            sum(1 for p in self.dataset_mapping.partitions if p.processing_status == "completed")
            if self.dataset_mapping and self.dataset_mapping.partitions
            else 0
        )
        failed_partitions = (
            sum(1 for p in self.dataset_mapping.partitions if p.processing_status == "failed")
            if self.dataset_mapping and self.dataset_mapping.partitions
            else 0
        )
        processing_partitions = (
            sum(1 for p in self.dataset_mapping.partitions if p.processing_status == "processing")
            if self.dataset_mapping and self.dataset_mapping.partitions
            else 0
        )
        total_partitions = (
            len(self.dataset_mapping.partitions) if self.dataset_mapping and self.dataset_mapping.partitions else 0
        )

        return {
            "total_partitions": total_partitions,
            "completed_partitions": completed_partitions,
            "failed_partitions": failed_partitions,
            "processing_partitions": processing_partitions,
        }

    def _calculate_checksum(self, data: List[Dict]) -> str:
        """Calculate checksum for partition data."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def _ensure_directory_exists(self, file_path: str):
        """Ensure the directory for a file path exists before writing."""
        # For Ray's write_parquet(), we need to create the entire path structure
        # because Ray creates a directory with the .parquet extension and puts files inside
        if file_path.endswith(".parquet"):
            # For parquet files, create the entire path as a directory
            # Also ensure all parent directories exist
            os.makedirs(file_path, exist_ok=True)
            # Create parent directories as well to ensure full path exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        else:
            # For other formats, just create the parent directory
            directory = os.path.dirname(file_path)
            if directory:  # Only create if there's a directory component
                os.makedirs(directory, exist_ok=True)

    @staticmethod
    @ray.remote
    def _ensure_directory_on_worker(file_path: str):
        """Remote function to ensure directory exists on Ray worker."""
        import os

        if file_path.endswith(".parquet"):
            # For parquet files, create the entire path as a directory
            os.makedirs(file_path, exist_ok=True)
            # Create parent directories as well
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        else:
            # For other formats, just create the parent directory
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        return True

    def _write_dataset_with_directory_creation(self, dataset, file_path: str, format_type: str = "parquet"):
        """Write dataset to file with automatic directory creation."""
        # Use absolute path for consistency
        abs_file_path = os.path.abspath(file_path)

        # Ensure directory exists both locally and on Ray workers
        self._ensure_directory_exists(abs_file_path)

        # Also ensure directory exists on Ray workers using remote function
        try:
            ray.get(self._ensure_directory_on_worker.remote(abs_file_path))
        except Exception as e:
            logger.warning(f"Failed to ensure directory on Ray worker: {e}")

        # Handle RayDataset objects by accessing the underlying Ray dataset
        if hasattr(dataset, "data"):
            # This is a RayDataset wrapper, use the underlying Ray dataset
            ray_dataset = dataset.data
        else:
            # This is a raw Ray dataset
            ray_dataset = dataset

        if format_type == "parquet":
            # For parquet, Ray creates a directory with the .parquet extension
            # and puts the actual parquet files inside that directory
            # Write as a single parquet file by setting a very high row limit
            ray_dataset.write_parquet(
                abs_file_path,
                num_rows_per_file=1000000,  # Large number to ensure single file
                compression=self.storage_compression,
            )
        elif format_type == "arrow":
            # OPTIMIZATION: Use Ray's built-in Arrow writing instead of materializing to Arrow table
            # This eliminates the memory overhead of converting to Arrow format
            # and keeps data distributed throughout the process
            ray_dataset.write_arrow(abs_file_path)
        else:  # jsonl
            ray_dataset.write_json(abs_file_path, force_ascii=False)

    def _estimate_partition_count(self, dataset) -> int:
        """Estimate the number of partitions based on dataset size."""
        try:
            total_samples = dataset.data.count()
            # Simple heuristic: 1 partition per 50k samples, minimum 1, maximum 10
            return max(1, min(10, total_samples // 50000))
        except Exception:
            # Fallback to default
            return 1

    def _can_use_natural_files(self, dataset) -> bool:
        """
        Check if we can use natural files as partitions.

        Returns True if we can extract original file paths from the dataset.
        """
        try:
            ray_dataset = dataset.data
            original_file_paths = self._extract_original_file_paths(ray_dataset)

            if len(original_file_paths) > 0:
                logger.info(f"Found {len(original_file_paths)} original files - can use natural file partitioning")
                return True
            else:
                logger.info("No original files found - will use manual partitioning")
                return False

        except Exception as e:
            logger.warning(f"Could not check for natural files: {e}")
            return False

    def _use_original_files_as_partitions(self, dataset) -> Tuple[List[str], DatasetMapping]:
        """
        Use original files as partitions when they are already well-sized.

        This method extracts the original file paths from the Ray dataset and uses them
        directly as partitions, avoiding the need to split and re-write data.
        """
        logger.info("Using original files as natural partitions...")

        try:
            # Get the underlying Ray dataset
            ray_dataset = dataset.data
            total_samples = ray_dataset.count()

            # Extract original file paths from Ray dataset metadata
            original_file_paths = self._extract_original_file_paths(ray_dataset)

            if not original_file_paths:
                logger.warning("Could not extract original file paths, falling back to single partition")
                return self._create_single_partition_fallback(dataset)

            logger.info(f"Found {len(original_file_paths)} original files to use as partitions")

            # Create partition metadata for each original file
            partitions = []
            for i, file_path in enumerate(original_file_paths):
                try:
                    # Get file size
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

                    # Estimate sample count for this file (rough approximation)
                    estimated_samples = max(1, file_size // 512)  # Assume ~512 bytes per sample

                    partition_metadata = PartitionMetadata(
                        partition_id=i,
                        original_start_idx=file_path,  # Store the file path here
                        original_end_idx=estimated_samples,
                        sample_count=estimated_samples,
                        file_size_bytes=file_size,
                        checksum="",  # Will be calculated if needed
                        created_timestamp=time.time(),
                        processing_status="pending",
                    )
                    partitions.append(partition_metadata)

                    logger.debug(f"Partition {i}: {file_path} ({file_size} bytes, ~{estimated_samples} samples)")

                except Exception as e:
                    logger.warning(f"Could not process file {file_path}: {e}")
                    continue

            if not partitions:
                logger.warning("No valid partitions found, falling back to single partition")
                return self._create_single_partition_fallback(dataset)

            # Create dataset mapping
            self.dataset_mapping = DatasetMapping(
                original_dataset_path=self.cfg.dataset_path,
                original_dataset_size=total_samples,
                partition_count=len(partitions),
                partition_size=sum(p.sample_count for p in partitions) // len(partitions),
                partitions=partitions,
            )

            logger.info(f"Successfully created {len(partitions)} partitions from original files")
            logger.info(f"Total estimated samples: {sum(p.sample_count for p in partitions)}")

            return original_file_paths, self.dataset_mapping

        except Exception as e:
            logger.error(f"Error using original files as partitions: {e}")
            logger.info("Falling back to single partition approach")
            return self._create_single_partition_fallback(dataset)

    def _extract_original_file_paths(self, ray_dataset) -> List[str]:
        """
        Extract original file paths from Ray dataset metadata.

        This method tries multiple approaches to get the original file paths
        that were used to create the Ray dataset.
        """
        file_paths = []

        try:
            # Method 1: Try to access Ray's internal metadata (for materialized datasets)
            if hasattr(ray_dataset, "_plan") and hasattr(ray_dataset._plan, "_in_blocks"):
                metadata = ray_dataset._plan._in_blocks
                if metadata:
                    for block_ref in metadata:
                        try:
                            block_metadata = ray.get(block_ref.get_metadata.remote())
                            if hasattr(block_metadata, "input_files"):
                                file_paths.extend(block_metadata.input_files)
                        except Exception:
                            continue

            # Method 2: Try to get file paths from dataset metadata (for materialized datasets)
            if not file_paths and hasattr(ray_dataset, "metadata"):
                try:
                    metadata = ray_dataset.metadata()
                    if metadata:
                        for block_metadata in metadata:
                            if hasattr(block_metadata, "input_files"):
                                file_paths.extend(block_metadata.input_files)
                except Exception:
                    pass

            # Method 3: Fallback - try to infer from dataset path (most reliable for directory-based datasets)
            if not file_paths:
                dataset_path = self.cfg.dataset_path
                if os.path.isfile(dataset_path):
                    file_paths = [dataset_path]
                elif os.path.isdir(dataset_path):
                    # Look for common data files in the directory
                    import glob

                    for ext in [".parquet", ".arrow", ".jsonl", ".json"]:
                        pattern = os.path.join(dataset_path, f"*{ext}")
                        found_files = glob.glob(pattern)
                        if found_files:
                            # Sort files to ensure consistent ordering
                            found_files.sort()
                            file_paths.extend(found_files)
                            logger.debug(f"Found {len(found_files)} {ext} files in directory: {found_files}")
                            break

            # Filter out non-existent files and remove duplicates
            file_paths = list(set([fp for fp in file_paths if os.path.exists(fp)]))

            logger.debug(f"Extracted {len(file_paths)} original file paths: {file_paths}")
            return file_paths

        except Exception as e:
            logger.debug(f"Error extracting original file paths: {e}")
            return []

    def _create_single_partition_fallback(self, dataset) -> Tuple[List[str], DatasetMapping]:
        """Fallback method when we can't use original files as partitions."""
        original_dataset_path = self.cfg.dataset_path
        total_samples = dataset.data.count()

        self.dataset_mapping = DatasetMapping(
            original_dataset_path=original_dataset_path,
            original_dataset_size=total_samples,
            partition_count=1,
            partition_size=total_samples,
        )

        # Return the original dataset path as the single partition
        return [original_dataset_path], self.dataset_mapping

    def _create_partitions_with_mapping(self, dataset) -> Tuple[List[str], DatasetMapping]:
        """Create partitions from the dataset with preserved mapping."""
        logger.info("Creating dataset partitions with mapping...")

        # Determine partitioning method
        partition_method = "natural_file"  # Default
        if hasattr(self.cfg, "partition") and self.cfg.partition:
            partition_method = getattr(self.cfg.partition, "method", "natural_file")

        logger.info(f"Partition method: {partition_method}")

        # Choose partitioning strategy based on method
        if partition_method == "natural_file":
            if self._can_use_natural_files(dataset):
                logger.info("Using natural files as partitions")
                return self._use_original_files_as_partitions(dataset)
            else:
                logger.warning("No natural files found, falling back to manual partitioning")
                # Fall through to manual partitioning
        elif partition_method == "optimized":
            raise NotImplementedError("Optimized partitioning not yet implemented")
        elif partition_method == "manual":
            raise NotImplementedError("Manual partitioning not yet implemented")
        else:
            logger.warning(f"Unknown partition method '{partition_method}', using natural files")
            if self._can_use_natural_files(dataset):
                return self._use_original_files_as_partitions(dataset)

        # If we reach here, natural file partitioning failed and manual is not implemented
        raise NotImplementedError("No natural files found and manual partitioning is not implemented")

    def _save_dataset_mapping(self):
        """Save dataset mapping to disk."""
        if self.dataset_mapping:
            mapping_path = os.path.join(self.metadata_dir, "dataset_mapping.json")
            with open(mapping_path, "w") as f:
                json.dump(asdict(self.dataset_mapping), f, indent=2, default=str)
            logger.info(f"Saved dataset mapping to {mapping_path}")

    def _load_dataset_mapping(self) -> Optional[DatasetMapping]:
        """Load dataset mapping from disk."""
        mapping_path = os.path.join(self.metadata_dir, "dataset_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping_data = json.load(f)

            # Reconstruct partition metadata
            partitions = []
            for p_data in mapping_data.get("partitions", []):
                partition = PartitionMetadata(**p_data)
                partitions.append(partition)

            mapping_data["partitions"] = partitions
            return DatasetMapping(**mapping_data)
        return None

    def _find_latest_operation_checkpoint(self, partition_id: int) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the latest operation checkpoint for a partition.

        Returns:
            Tuple of (latest_op_idx, checkpoint_path) or (None, None) if no checkpoint found
        """
        partition_checkpoint_dir = os.path.join(self.checkpoint_dir, f"partition_{partition_id:06d}")

        if not os.path.exists(partition_checkpoint_dir):
            return None, None

        # Find all operation checkpoint directories (Ray creates directories, not files)
        checkpoint_dirs = []
        for item in os.listdir(partition_checkpoint_dir):
            item_path = os.path.join(partition_checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("op_") and item.endswith(f".{self.storage_format}"):
                try:
                    # Extract operation index from directory name: op_XXX_OpName.parquet
                    op_idx = int(item.split("_")[1])
                    checkpoint_dirs.append((op_idx, item))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_dirs:
            return None, None

        # Return the latest operation checkpoint
        latest_op_idx, latest_dir = max(checkpoint_dirs, key=lambda x: x[0])
        checkpoint_path = os.path.join(partition_checkpoint_dir, latest_dir)

        logger.info(f"Found operation checkpoint for partition {partition_id}: op_{latest_op_idx} at {checkpoint_path}")
        return latest_op_idx, checkpoint_path

    def _load_operation_checkpoint(self, checkpoint_path: str) -> ray.data.Dataset:
        """Load dataset from operation checkpoint."""
        # Convert relative path to absolute path to avoid Ray path resolution issues
        checkpoint_path = os.path.abspath(checkpoint_path)

        if self.storage_format == "parquet":
            return ray.data.read_parquet(checkpoint_path)
        elif self.storage_format == "arrow":
            import pyarrow.feather as feather

            table = feather.read_feather(checkpoint_path)
            if hasattr(table, "to_pandas"):
                df = table.to_pandas()
            else:
                df = table
            return ray.data.from_pandas(df)
        else:  # jsonl
            return ray.data.read_json(checkpoint_path)

    def _process_partition(self, partition_path: str, ops: List, partition_id: int) -> Dict[str, Any]:
        """Process a single partition with all operations."""
        partition_start_time = time.time()

        try:
            # Log partition start
            partition_meta = {
                "partition_path": partition_path,
                "partition_id": partition_id,
                "start_time": partition_start_time,
            }
            self.log_partition_start(partition_id, partition_meta)

            # Update partition status to processing
            if self.dataset_mapping and partition_id < len(self.dataset_mapping.partitions):
                self.dataset_mapping.partitions[partition_id].processing_status = "processing"
                self.dataset_mapping.partitions[partition_id].processing_start_time = partition_start_time
                self._save_dataset_mapping()

            # Check for existing operation checkpoint
            latest_op_idx, latest_checkpoint_path = self._find_latest_operation_checkpoint(partition_id)
            if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
                logger.info(f"Loading checkpoint for partition {partition_id} from operation {latest_op_idx}")
                raw_dataset = self._load_operation_checkpoint(latest_checkpoint_path)

                current_dataset = RayDataset(raw_dataset, dataset_path=partition_path, cfg=self.cfg)
                ops_to_process = ops[latest_op_idx + 1 :]
            else:
                # Load partition data using DatasetBuilder for proper handling
                # Convert relative path to absolute path to avoid Ray path resolution issues
                partition_path = os.path.abspath(partition_path)
                logger.debug(f"Loading partition {partition_id} from {partition_path}")

                # Create DatasetBuilder instance for this partition
                from data_juicer.core.data.dataset_builder import DatasetBuilder

                partition_cfg = Namespace(**vars(self.cfg))
                partition_cfg.dataset_path = partition_path  # Override dataset_path for this partition
                datasetbuilder = DatasetBuilder(partition_cfg, executor_type="ray")

                # Load dataset using DatasetBuilder for proper handling
                current_dataset = datasetbuilder.load_dataset()
                ops_to_process = ops

            # Create checkpoint directory for operation checkpoints
            partition_checkpoint_dir = os.path.join(self.checkpoint_dir, f"partition_{partition_id:06d}")
            os.makedirs(partition_checkpoint_dir, exist_ok=True)

            # Process all operations at once (like Ray executor) for performance
            input_rows = current_dataset.data.count()

            # Log all operations as starting (for compatibility with existing logging)
            for op_idx, op in enumerate(ops_to_process):
                actual_op_idx = latest_op_idx + 1 + op_idx if latest_op_idx is not None else op_idx
                self.log_op_start(partition_id, op._name, actual_op_idx, {})

            # Process ALL operations in a single Ray task (8x performance improvement)
            logger.info(f"Processing {len(ops_to_process)} operations on partition {partition_id} in batch")
            batch_start_time = time.time()
            current_dataset.process(ops_to_process)  # Process all operations at once!
            batch_duration = time.time() - batch_start_time

            # Get final row count
            output_rows = current_dataset.data.count()

            # Log all operations as completed (for compatibility with existing logging)
            for op_idx, op in enumerate(ops_to_process):
                actual_op_idx = latest_op_idx + 1 + op_idx if latest_op_idx is not None else op_idx
                # Distribute batch time across operations for logging
                op_duration = batch_duration / len(ops_to_process)

                self.log_op_complete(
                    partition_id,
                    op._name,
                    actual_op_idx,
                    op_duration,
                    None,  # No individual operation checkpoints
                    input_rows if op_idx == 0 else input_rows,  # Use input_rows for all ops to avoid None subtraction
                    (
                        output_rows if op_idx == len(ops_to_process) - 1 else input_rows
                    ),  # Use input_rows for intermediate ops
                )

            # Create partition-level checkpoint (instead of operation-level)
            if self.checkpoint_enabled:
                partition_checkpoint_path = os.path.join(
                    partition_checkpoint_dir, f"partition_{partition_id:06d}_completed.parquet"
                )
                self._write_dataset_with_directory_creation(current_dataset, partition_checkpoint_path, "parquet")
                logger.debug(f"Saved partition-level checkpoint for partition {partition_id}")

            # Write final output
            output_path = os.path.join(self.results_dir, f"partition_{partition_id:06d}.parquet")
            self._write_dataset_with_directory_creation(current_dataset, output_path, "parquet")

            # Create checkpoint at partition completion if strategy is "every_partition"
            if self.checkpoint_strategy == CheckpointStrategy.EVERY_PARTITION and self.checkpoint_enabled:
                partition_checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"partition_{partition_id:06d}_final.parquet"
                )
                self._write_dataset_with_directory_creation(
                    current_dataset, partition_checkpoint_path, "parquet"  # Always use parquet for final checkpoints
                )

                # Log checkpoint event
                self._log_processing_event(
                    ProcessingEvent(
                        event_id=f"partition_checkpoint_{partition_id}_{int(time.time())}",
                        event_type="partition_checkpoint",
                        timestamp=time.time(),
                        partition_id=partition_id,
                        message=f"Created final checkpoint for partition {partition_id}",
                        metadata={"checkpoint_path": partition_checkpoint_path},
                    )
                )

                logger.debug(f"Created partition checkpoint: {partition_checkpoint_path}")

            # Update partition status
            if self.dataset_mapping and partition_id < len(self.dataset_mapping.partitions):
                self.dataset_mapping.partitions[partition_id].processing_status = "completed"
                self.dataset_mapping.partitions[partition_id].processing_end_time = time.time()
                self._save_dataset_mapping()

            # Log partition completion with debug information
            partition_duration = time.time() - partition_start_time
            logger.debug(f"Partition {partition_id} completed successfully in {partition_duration:.2f}s")
            logger.debug(f"Calling log_partition_complete for partition {partition_id}")
            self.log_partition_complete(partition_id, partition_duration, output_path)
            logger.debug(f"Successfully logged partition_complete event for partition {partition_id}")

            return {
                "partition_id": partition_id,
                "input_path": partition_path,
                "output_path": output_path,
                "checkpoint_dir": partition_checkpoint_dir,
                "success": True,
                "sample_count": current_dataset.data.count(),
                "processing_time": time.time()
                - (self.dataset_mapping.partitions[partition_id].processing_start_time or time.time()),
            }

        except Exception as e:
            logger.error(f"Error processing partition {partition_id}: {e}")

            # Update partition status
            if self.dataset_mapping and partition_id < len(self.dataset_mapping.partitions):
                self.dataset_mapping.partitions[partition_id].processing_status = "failed"
                self.dataset_mapping.partitions[partition_id].error_message = str(e)
                self.dataset_mapping.partitions[partition_id].retry_count += 1
                self._save_dataset_mapping()

            # Log partition failure
            partition_duration = time.time() - partition_start_time
            self.log_partition_failed(partition_id, str(e), 0)

            # Also log partition completion (failure case)
            logger.debug(f"Logging partition_complete event for failed partition {partition_id}")
            self.log_partition_complete(partition_id, partition_duration, None, success=False, error=str(e))
            logger.debug(f"Successfully logged partition_complete event for failed partition {partition_id}")

            raise

    def _process_partition_with_retry(self, partition_path: str, ops: List, partition_id: int) -> Dict[str, Any]:
        """Process partition with retry logic for fault tolerance."""
        partition_start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                return self._process_partition(partition_path, ops, partition_id)
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed for partition {partition_id}: {e}")

                    # Calculate exponential backoff delay
                    delay = 2**attempt

                    logger.info(f"Retrying partition {partition_id} in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All attempts failed for partition {partition_id}: {e}")

                    # Log final partition completion (failure after all retries)
                    partition_duration = time.time() - partition_start_time
                    logger.debug(
                        f"Logging final partition_complete event for failed partition {partition_id} after all retries"
                    )
                    self.log_partition_complete(partition_id, partition_duration, None, success=False, error=str(e))
                    logger.debug(
                        f"Successfully logged final partition_complete event for failed partition {partition_id}"
                    )

                    return {
                        "partition_id": partition_id,
                        "input_path": partition_path,
                        "output_path": None,
                        "success": False,
                        "error": str(e),
                        "retry_count": attempt + 1,
                    }

    def _merge_partitions_with_mapping(self, partition_results: List[Dict[str, Any]]) -> str:
        """Write processed partitions directly to final output directory (no merging)."""
        logger.info("Writing partitions directly to final output directory...")

        # Filter successful partitions
        successful_results = [r for r in partition_results if r["success"]]
        failed_results = [r for r in partition_results if not r["success"]]

        if failed_results:
            logger.warning(f"{len(failed_results)} partitions failed processing")
            for failed in failed_results:
                logger.warning(f"Partition {failed['partition_id']} failed: {failed.get('error', 'Unknown error')}")

        if not successful_results:
            raise RuntimeError("No partitions were processed successfully")

        # Sort partitions by ID to maintain original order
        successful_results.sort(key=lambda x: x["partition_id"])

        # Use standard exporter to combine all partitions
        logger.info("Creating combined dataset from all partitions...")
        result_path = self._export_combined_dataset(successful_results)

        # Create final mapping report
        self._create_final_mapping_report(partition_results)

        logger.info(f"Successfully exported {len(successful_results)} partitions")
        return result_path

    def _export_combined_dataset(self, successful_results: List[Dict[str, Any]]) -> str:
        """Export all partitions combined into a single dataset (original behavior)."""
        logger.info("Creating combined dataset from all partitions...")

        # Load all partition datasets
        partition_datasets = []
        total_samples = 0

        for result in successful_results:
            if result["output_path"] and os.path.exists(result["output_path"]):
                try:
                    # Convert relative path to absolute path to avoid Ray path resolution issues
                    output_path = os.path.abspath(result["output_path"])
                    partition_id = result["partition_id"]

                    logger.info(f"Loading partition {partition_id} from {output_path}")

                    # Load the partition dataset
                    if os.path.isdir(output_path):
                        logger.info(f"Partition {partition_id} is a directory, reading as parquet dataset")
                        partition_dataset = ray.data.read_parquet(output_path)
                    elif output_path.endswith(".parquet"):
                        logger.info(f"Partition {partition_id} is a parquet file")
                        partition_dataset = ray.data.read_parquet(output_path)
                    elif output_path.endswith(".arrow"):
                        logger.info(f"Partition {partition_id} is an arrow file")
                        partition_dataset = ray.data.read_arrow(output_path)
                    else:
                        logger.info(f"Partition {partition_id} is assumed to be JSONL")
                        partition_dataset = ray.data.read_json(output_path)

                    # Get sample count
                    sample_count = partition_dataset.count()
                    total_samples += sample_count
                    logger.info(f"Partition {partition_id} has {sample_count} samples")
                    partition_datasets.append(partition_dataset)

                except Exception as e:
                    logger.error(f"Failed to load partition {result['partition_id']}: {e}")
                    continue

        if not partition_datasets:
            raise RuntimeError("No partition datasets could be loaded successfully")

        # Combine all partitions into one dataset (this doesn't merge, just combines)
        logger.info(f"Combining {len(partition_datasets)} partitions...")
        combined_dataset = partition_datasets[0]
        for i, partition_dataset in enumerate(partition_datasets[1:], 1):
            logger.info(f"Combining partition {i+1}/{len(partition_datasets)}...")
            combined_dataset = combined_dataset.union(partition_dataset)

        # Write the combined dataset directly - this will create fragmented output
        logger.info("Writing combined dataset to final output (will create fragmented files)...")
        self.exporter.export(combined_dataset, columns=None)
        logger.info(f"Successfully exported combined dataset with {total_samples} total samples")

        return self.cfg.export_path

    def _create_final_mapping_report(self, partition_results: List[Dict[str, Any]]):
        """Create a final mapping report showing the relationship between original and processed data."""
        if not self.dataset_mapping:
            return

        report = {
            "original_dataset": {
                "path": self.dataset_mapping.original_dataset_path,
                "total_samples": self.dataset_mapping.original_dataset_size,
                "partition_count": self.dataset_mapping.partition_count,
            },
            "processing_summary": {
                "total_partitions": len(partition_results),
                "successful_partitions": len([r for r in partition_results if r["success"]]),
                "failed_partitions": len([r for r in partition_results if not r["success"]]),
                "total_processed_samples": sum(r.get("sample_count", 0) for r in partition_results if r["success"]),
            },
            "partition_details": [],
        }

        for result in partition_results:
            partition_id = result["partition_id"]
            if partition_id < len(self.dataset_mapping.partitions):
                partition_meta = self.dataset_mapping.partitions[partition_id]
                report["partition_details"].append(
                    {
                        "partition_id": partition_id,
                        "original_range": f"{partition_meta.original_start_idx}-{partition_meta.original_end_idx}",
                        "original_samples": partition_meta.sample_count,
                        "processed_samples": result.get("sample_count", 0),
                        "status": result["success"],
                        "processing_time": result.get("processing_time", 0),
                        "error": result.get("error", None),
                    }
                )

        # Save final report
        report_path = os.path.join(self.metadata_dir, "final_mapping_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Created final mapping report: {report_path}")

    def _get_operation_progress(self) -> Dict[int, int]:
        """Get operation progress for each partition."""
        progress = {}
        for partition_id in range(self.dataset_mapping.partition_count if self.dataset_mapping else 0):
            latest_op_idx, _ = self._find_latest_operation_checkpoint(partition_id)
            progress[partition_id] = latest_op_idx + 1 if latest_op_idx is not None else 0
        return progress

    def _save_checkpoint(self, partition_results: List[Dict[str, Any]], ops: List) -> str:
        """Save processing checkpoint with enhanced metadata."""
        checkpoint_data = {
            "timestamp": time.time(),
            "partition_results": partition_results,
            "ops_completed": len(ops),
            "total_partitions": len(partition_results),
            "dataset_mapping": asdict(self.dataset_mapping) if self.dataset_mapping else None,
            "operation_progress": self._get_operation_progress(),
        }

        # Create checkpoint filename with timestamp
        checkpoint_filename = f"checkpoint_{int(time.time())}.json"
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_filename)

        # Ensure checkpoint directory exists
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

        # Save checkpoint data
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint if available."""
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]
        if not checkpoint_files:
            return None

        # Get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)

        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

            # Reconstruct the DatasetMapping object if it exists
            if checkpoint_data.get("dataset_mapping"):
                mapping_data = checkpoint_data["dataset_mapping"]
                # Reconstruct partitions as PartitionMetadata objects
                if mapping_data.get("partitions"):
                    partitions = []
                    for partition_data in mapping_data["partitions"]:
                        partition = PartitionMetadata(**partition_data)
                        partitions.append(partition)
                    mapping_data["partitions"] = partitions

                # Reconstruct the DatasetMapping object
                checkpoint_data["dataset_mapping"] = DatasetMapping(**mapping_data)

            return checkpoint_data

    def _create_job_summary(self, job_id: str, job_dir: str):
        """Create and display job summary for easy resumption."""
        # Use already-resolved paths from config
        job_summary = {
            "job_id": job_id,
            "start_time": time.time(),
            "work_dir": self.work_dir,
            "job_dir": job_dir,
            "config_file": getattr(self.cfg, "config", None),
            "backed_up_config_path": getattr(self.cfg, "backed_up_config_path", None),
            "executor_type": getattr(self, "executor_type", "unknown"),
            "status": "running",
            "resumption_command": f"dj-process --config {getattr(self.cfg, 'config', 'config.yaml')} --job_id {job_id}",
            "event_log_file": self.cfg.event_log_file,
            "event_log_dir": self.cfg.event_log_dir,
            "checkpoint_dir": self.cfg.checkpoint_dir,
            "metadata_dir": self.cfg.metadata_dir,
        }

        # Write job summary to the already-resolved job_summary_file path
        with open(self.cfg.job_summary_file, "w") as f:
            json.dump(job_summary, f, indent=2, default=str)

        logger.info("=" * 60)
        logger.info("DataJuicer Job Started")
        logger.info("=" * 60)
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Job Directory: {job_dir}")
        logger.info(f"Work Directory: {self.work_dir}")
        logger.info(f"Event Logs: {job_summary['event_log_file']}")
        logger.info(f"Checkpoints: {self.cfg.checkpoint_dir}")
        logger.info(f"Event Log Storage: {self.cfg.event_log_dir}")
        logger.info(f"Checkpoint Storage: {self.cfg.checkpoint_dir}")
        logger.info("=" * 60)
        logger.info("To resume this job later, use:")
        logger.info(f"  {job_summary['resumption_command']}")
        logger.info("=" * 60)

    def _get_config_content(self, config_path) -> Optional[str]:
        """Get config file content, handling different path types."""
        try:
            if isinstance(config_path, list) and config_path:
                config_path = config_path[0]
            path_str = str(config_path)
            if os.path.exists(path_str):
                with open(path_str, "r") as f:
                    return f.read()
        except Exception:
            pass
        return None

    def _compare_config_contents(self, current_content: str, saved_content: str) -> Dict[str, Any]:
        """Compare config file contents."""
        return {
            "configs_match": current_content == saved_content,
            "can_resume": current_content == saved_content,
            "reason": None if current_content == saved_content else "Config contents differ",
        }

    def _validate_job_resumption(self, job_id: str) -> Dict[str, Any]:
        """
        Enhanced job resumption validation using event analysis and config file comparison.

        Args:
            job_id: The job ID to validate for resumption

        Returns:
            Dictionary containing resumption validation results and plan
        """
        logger.info(f"Validating job resumption for job_id: {job_id}")

        # Check if job directory exists
        job_dir = Path(self.cfg.work_dir).parent / job_id
        if not job_dir.exists():
            logger.warning(f"Job directory not found: {job_dir}")
            return {"can_resume": False, "reason": "Job directory not found"}

        # Check if job summary exists
        job_summary_file = job_dir / "job_summary.json"
        if not job_summary_file.exists():
            logger.warning(f"Job summary not found: {job_summary_file}")
            return {"can_resume": False, "reason": "Job summary not found"}

        # Load job summary
        try:
            with open(job_summary_file, "r") as f:
                job_summary = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load job summary: {e}")
            return {"can_resume": False, "reason": f"Failed to load job summary: {e}"}

        # Validate config file compatibility
        current_config = getattr(self.cfg, "config", None)
        backed_up_config_path = getattr(self.cfg, "backed_up_config_path", None)

        if current_config and backed_up_config_path and os.path.exists(backed_up_config_path):
            # Get the actual config content for comparison
            current_content = self._get_config_content(current_config)
            saved_content = self._get_config_content(backed_up_config_path)

            if current_content and saved_content:
                config_validation = self._compare_config_contents(current_content, saved_content)
                if not config_validation["can_resume"]:
                    logger.error(f"❌ Config validation failed: {config_validation['reason']}")
                    return {
                        "can_resume": False,
                        "reason": config_validation["reason"],
                        "config_validation": config_validation,
                        "job_id": job_id,
                        "job_dir": str(job_dir),
                        "validation_timestamp": time.time(),
                    }
            else:
                config_validation = {
                    "configs_match": True,
                    "can_resume": True,
                    "reason": "Config content not available",
                }
        else:
            config_validation = {"configs_match": True, "can_resume": True, "reason": "Config comparison skipped"}

        # Analyze resumption state using events
        resumption_analysis = self.analyze_resumption_state(job_id)

        if "error" in resumption_analysis:
            logger.warning(f"Resumption analysis failed: {resumption_analysis['error']}")
            return {"can_resume": False, "reason": resumption_analysis["error"]}

        # Combine job summary with resumption analysis and config validation
        validation_result = {
            "can_resume": resumption_analysis["can_resume"] and config_validation["can_resume"],
            "job_id": job_id,
            "job_dir": str(job_dir),
            "job_summary": job_summary,
            "resumption_analysis": resumption_analysis,
            "config_validation": config_validation,
            "validation_timestamp": time.time(),
        }

        if validation_result["can_resume"]:
            logger.info(f"✅ Job resumption validated successfully")
            logger.info(f"   Job status: {resumption_analysis['job_status']}")
            logger.info(f"   Partitions to retry: {resumption_analysis['partitions_to_retry']}")
            logger.info(f"   Partitions to skip: {resumption_analysis['partitions_to_skip']}")
            logger.info(f"   Progress: {resumption_analysis['progress_metrics']['progress_percentage']:.1f}%")
            logger.info(f"   Config compatibility: ✅ Valid")

            if resumption_analysis.get("resume_from_checkpoint"):
                logger.info(f"   Resume from checkpoint: {resumption_analysis['resume_from_checkpoint']}")
        else:
            logger.warning(f"❌ Job resumption validation failed")
            if not config_validation["can_resume"]:
                logger.warning(f"   Config reason: {config_validation.get('reason', 'Unknown')}")
            if not resumption_analysis["can_resume"]:
                logger.warning(f"   Resumption reason: {resumption_analysis.get('reason', 'Unknown')}")

        return validation_result

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Run the partitioned dataset processing pipeline.

        Args:
            load_data_np: Number of workers for loading dataset
            skip_return: Whether to skip returning the dataset

        Returns:
            Processed dataset
        """
        job_start_time = time.time()

        # Check if this is a resumption attempt
        # Only attempt resumption if job_id was explicitly provided by user (not auto-generated)
        # We can detect this by checking if the job_id was set before the config was processed
        user_provided_job_id = getattr(self.cfg, "_user_provided_job_id", False)

        if user_provided_job_id and hasattr(self.cfg, "job_id") and self.cfg.job_id:
            logger.info(f"🔍 Checking for job resumption: {self.cfg.job_id}")

            # Validate resumption using event analysis
            resumption_validation = self._validate_job_resumption(self.cfg.job_id)

            if resumption_validation["can_resume"]:
                logger.info(f"🔄 Resuming job: {self.cfg.job_id}")
                return self._resume_job(resumption_validation)
            else:
                # Extract reason from the nested resumption_analysis -> resumption_plan
                resumption_analysis = resumption_validation.get("resumption_analysis", {})
                resumption_plan = resumption_analysis.get("resumption_plan", {})
                reason = resumption_plan.get("reason", "Unknown")

                logger.error(f"❌ Job resumption failed: {reason}")
                logger.error(f"   Cannot resume job {self.cfg.job_id} with the same job_id")
                logger.error(f"   Please use a different job_id or fix the validation issues")
                raise RuntimeError(f"Job resumption failed: {reason}")
        else:
            logger.info(f"🚀 Starting new job with job_id: {getattr(self.cfg, 'job_id', 'auto-generated')}")

        # Create job summary at the start of the run
        self._create_job_summary(self.cfg.job_id, self.cfg.job_dir)

        # Extract file paths directly (skip dataset loading for efficiency)
        logger.info("Extracting file paths from dataset...")
        file_paths = self._extract_file_paths_from_config()

        # Prepare process operators
        logger.info("Preparing process operators...")
        ops = self._prepare_operators()

        # Create partitions from files (much more efficient)
        logger.info("Creating partitions from file paths...")

        # Log repartition start event
        repartition_start_time = time.time()
        self._log_processing_event(
            ProcessingEvent(
                event_id=f"repartition_start_{int(repartition_start_time)}",
                event_type="repartition_start",
                timestamp=repartition_start_time,
                message="Starting file-based partitioning phase",
                metadata={
                    "original_dataset_path": self.cfg.dataset_path,
                    "file_count": len(file_paths),
                    "storage_format": (
                        getattr(self.cfg.intermediate_storage, "format", "parquet")
                        if hasattr(self.cfg, "intermediate_storage")
                        else "parquet"
                    ),
                    "compression": (
                        getattr(self.cfg.intermediate_storage, "compression", "snappy")
                        if hasattr(self.cfg, "intermediate_storage")
                        else "snappy"
                    ),
                },
            )
        )

        partition_paths, self.dataset_mapping = self._create_partitions_from_files(file_paths)

        # Log repartition complete event
        repartition_complete_time = time.time()
        repartition_duration = repartition_complete_time - repartition_start_time
        self._log_processing_event(
            ProcessingEvent(
                event_id=f"repartition_complete_{int(repartition_complete_time)}",
                event_type="repartition_complete",
                timestamp=repartition_complete_time,
                message=f"Dataset repartitioning completed - {len(partition_paths)} partitions created in {repartition_duration:.2f}s",
                metadata={
                    "partition_count": len(partition_paths),
                    "total_samples": self.dataset_mapping.original_dataset_size,
                    "partition_paths": partition_paths,
                    "duration_seconds": repartition_duration,
                    "partitions_dir": self.partitions_dir,
                },
            )
        )

        # Initialize DAG execution planning AFTER partitioning to use actual partition count
        logger.info(f"Initializing DAG execution planning with {len(partition_paths)} actual partitions...")
        self._initialize_dag_execution(self.cfg)

        # Log job start with DAG context
        job_config = {
            "dataset_path": self.cfg.dataset_path,
            "work_dir": self.work_dir,
            "executor_type": self.executor_type,
            "dag_node_count": len(self.pipeline_dag.nodes) if self.pipeline_dag else 0,
            "dag_edge_count": len(self.pipeline_dag.edges) if self.pipeline_dag else 0,
            "parallel_groups_count": len(self.pipeline_dag.parallel_groups) if self.pipeline_dag else 0,
        }
        self.log_job_start(job_config, len(ops))

        # Process files in parallel using Ray actors (much more efficient)
        logger.info(f"Processing {len(partition_paths)} files in parallel...")

        partition_results = self._process_files_parallel(partition_paths, ops)

        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        self._save_checkpoint(partition_results, ops)

        # Export final dataset using RayExporter (same as ray_executor.py)
        logger.info("Exporting dataset to disk...")

        if len(partition_results) == 1 and partition_results[0].get("input_path") == "merged":
            # Convergence was used, get the processed dataset
            logger.info("Convergence processing completed, exporting final dataset...")
            final_dataset = partition_results[0]["dataset"]
        else:
            # No convergence, load the processed dataset
            logger.info("Loading processed dataset...")
            final_dataset = self._load_processed_dataset_from_results(partition_results)

        if hasattr(final_dataset, "data"):
            # RayDataset wrapper
            self.exporter.export(final_dataset.data, columns=final_dataset.data.columns())
        else:
            # Raw Ray dataset
            self.exporter.export(final_dataset, columns=final_dataset.columns())
        final_output_path = self.cfg.export_path

        # Log job completion
        job_duration = time.time() - job_start_time
        self.log_job_complete(job_duration, final_output_path)

        logger.info(f"✅ Job completed successfully in {job_duration:.2f}s")
        logger.info(f"📁 Output saved to: {final_output_path}")

        if skip_return:
            return None

        # Return processed dataset
        return final_dataset

    def _resume_job(self, resumption_validation: Dict[str, Any]):
        """
        Resume a job based on resumption analysis.

        Args:
            resumption_validation: Validation result from _validate_job_resumption

        Returns:
            Processed dataset
        """
        logger.info(f"🔄 Resuming job with intelligent event-based resumption")

        resumption_analysis = resumption_validation["resumption_analysis"]

        # Determine what needs to be processed
        partitions_to_retry = resumption_analysis["partitions_to_retry"]
        partitions_to_skip = resumption_analysis["partitions_to_skip"]

        logger.info(f"📊 Resumption Plan:")
        logger.info(f"   Partitions to retry: {partitions_to_retry}")
        logger.info(f"   Partitions to skip: {partitions_to_skip}")
        logger.info(
            f"   Estimated remaining work: {resumption_analysis['resumption_plan']['estimated_remaining_work']*100:.1f}%"
        )

        # Prepare operators
        logger.info("Preparing process operators...")
        ops = self._prepare_operators()

        # Initialize DAG execution planning for resumption
        self._initialize_dag_execution(self.cfg)

        # Load existing partitions
        partition_paths = self._load_existing_partitions()

        # Process only partitions that need retrying in parallel using threads
        partition_results = []

        # Collect partitions that need processing
        partitions_to_process = []
        for i, partition_path in enumerate(partition_paths):
            if i in partitions_to_retry:
                logger.info(f"🔄 Partition {i} will be retried")
                partitions_to_process.append((i, partition_path, "retry"))
            elif i in partitions_to_skip:
                logger.info(f"⏭️  Skipping completed partition {i}")
                # Load the existing result for this partition
                existing_result = self._load_partition_result(i)
                partition_results.append(existing_result)
            else:
                logger.info(f"❓ Partition {i} will be processed normally")
                partitions_to_process.append((i, partition_path, "normal"))

        # Process partitions in parallel using ThreadPoolExecutor
        if partitions_to_process:
            with ThreadPoolExecutor(max_workers=min(len(partitions_to_process), 8)) as executor:
                # Submit partition processing tasks
                future_to_partition = {}
                for i, partition_path, task_type in partitions_to_process:
                    logger.info(f"Submitting partition {i} for {task_type} processing")
                    future = executor.submit(self._process_partition_with_retry, partition_path, ops, i)
                    future_to_partition[future] = (i, task_type)

                # Collect results as they complete
                for future in as_completed(future_to_partition):
                    partition_id, task_type = future_to_partition[future]
                    try:
                        result = future.result()
                        partition_results.append(result)
                        logger.info(f"Partition {partition_id} ({task_type}) completed successfully")
                    except Exception as e:
                        logger.error(f"Partition {partition_id} ({task_type}) failed with exception: {e}")
                        partition_results.append(
                            {
                                "partition_id": partition_id,
                                "input_path": partition_paths[partition_id],
                                "output_path": None,
                                "success": False,
                                "error": str(e),
                            }
                        )

        # Merge partitions
        logger.info("Merging processed partitions...")
        final_output_path = self._merge_partitions_with_mapping(partition_results)

        # Log job completion
        job_duration = time.time() - time.time()  # This should be calculated properly
        self.log_job_complete(job_duration, final_output_path)

        logger.info(f"✅ Job resumed and completed successfully")
        logger.info(f"📁 Output saved to: {final_output_path}")

        return self._load_processed_dataset(final_output_path)

    def _load_dataset(self, load_data_np: Optional[int] = None):
        """Load dataset using the dataset builder."""
        # Create DatasetBuilder instance for loading the full dataset
        datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")
        return datasetbuilder.load_dataset(num_proc=load_data_np)

    def _extract_file_paths_from_config(self) -> List[str]:
        """
        Extract file paths directly from config without loading the full dataset.
        """
        dataset_path = self.cfg.dataset_path

        if os.path.isfile(dataset_path):
            # Single file
            return [dataset_path]
        elif os.path.isdir(dataset_path):
            # Directory - find all supported files
            supported_extensions = [".json", ".jsonl", ".parquet", ".arrow"]
            file_paths = []

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if any(file.endswith(ext) for ext in supported_extensions):
                        file_paths.append(os.path.join(root, file))

            # Sort files for consistent ordering
            file_paths.sort()
            return file_paths
        else:
            raise ValueError(f"Dataset path not found: {dataset_path}")

    def _create_partitions_from_files(self, file_paths: List[str]) -> Tuple[List[str], DatasetMapping]:
        """
        Create partitions directly from file paths without loading dataset.

        This is much more efficient than loading the entire dataset first.
        """
        logger.info(f"Creating {len(file_paths)} partitions from file paths...")

        partitions = []
        total_size = 0

        for i, file_path in enumerate(file_paths):
            file_size = os.path.getsize(file_path)
            total_size += file_size

            # Calculate checksum for the file
            with open(file_path, "rb") as f:
                file_content = f.read()
                checksum = hashlib.md5(file_content).hexdigest()

            partition = PartitionMetadata(
                partition_id=i,
                original_start_idx=0,  # File-based, so start from 0
                original_end_idx=0,  # Will be updated after processing
                sample_count=0,  # Will be updated after processing
                file_size_bytes=file_size,
                checksum=checksum,
                created_timestamp=time.time(),
                file_path=file_path,
            )
            partitions.append(partition)

        dataset_mapping = DatasetMapping(
            original_dataset_path=self.cfg.dataset_path,
            original_dataset_size=total_size,
            partition_count=len(partitions),
            partition_size=total_size // len(partitions) if partitions else 0,
            partitions=partitions,
        )

        logger.info(f"Created {len(partitions)} partitions with total size {total_size / (1024*1024):.2f} MB")

        return file_paths, dataset_mapping

    def _process_files_parallel(self, file_paths: List[str], ops: List) -> List[Dict[str, Any]]:
        """
        Process files in parallel using Ray tasks with convergence support for global operations.

        This provides true parallelism by processing each file in its own Ray task,
        but handles global operations by converging partitions when needed.
        """
        logger.info(f"Processing {len(file_paths)} files in parallel using Ray tasks...")

        # Detect convergence points for global operations
        convergence_points = self._detect_convergence_points_partitioned(self.cfg)

        if convergence_points:
            logger.info(f"Found convergence points at operations: {convergence_points}")
            return self._process_with_convergence(file_paths, ops, convergence_points)
        else:
            logger.info("No convergence points found, processing files sequentially")
            return self._process_without_convergence(file_paths, ops)

    def _process_with_convergence(
        self, file_paths: List[str], ops: List, convergence_points: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Process files with convergence support for global operations.

        This method:
        1. Processes partitions up to the first convergence point
        2. Merges partitions for global operations
        3. Continues processing the merged dataset
        """
        logger.info("Processing with convergence support for global operations...")

        # Find the first convergence point
        first_convergence = min(convergence_points)
        logger.info(f"First convergence point at operation {first_convergence}")

        # Split operations into pre-convergence and post-convergence
        pre_convergence_ops = ops[:first_convergence]
        post_convergence_ops = ops[first_convergence:]

        logger.info(f"Pre-convergence operations: {len(pre_convergence_ops)}")
        logger.info(f"Post-convergence operations: {len(post_convergence_ops)}")

        # Process partitions up to convergence point
        if pre_convergence_ops:
            logger.info("Processing partitions up to convergence point...")
            partition_results = self._process_without_convergence(file_paths, pre_convergence_ops)
        else:
            logger.info("No pre-convergence operations, loading raw partitions...")
            partition_results = self._load_raw_partitions(file_paths)

        # Merge partitions for global operations, and materialize to avoid union chain
        logger.info("Merging partitions for global operations...")
        merged_dataset = self._merge_partitions_for_global_ops(partition_results)

        # Process merged dataset with post-convergence operations
        if post_convergence_ops:
            logger.info("Processing merged dataset with global operations...")
            final_dataset = self._process_merged_dataset(merged_dataset, post_convergence_ops)

            # Return the processed dataset for final export in run()
            return [
                {"partition_id": 0, "input_path": "merged", "dataset": final_dataset, "success": True, "error": None}
            ]
        else:
            # No post-convergence operations, just return the merged result
            return [
                {"partition_id": 0, "input_path": "merged", "dataset": merged_dataset, "success": True, "error": None}
            ]

    def _process_without_convergence(self, file_paths: List[str], ops: List) -> List[Dict[str, Any]]:
        """
        Process files without convergence using ThreadPoolExecutor for parallelism
        """
        logger.info("Processing files without convergence using ThreadPoolExecutor...")

        results = []
        max_workers = min(len(file_paths), 4)  # Limit to 4 workers to avoid overwhelming the system

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(_process_single_batch, file_path, ops, i, self.work_dir, self.cfg): i
                for i, file_path in enumerate(file_paths)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                file_path = file_paths[i]

                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"File {i+1} processed successfully")
                except Exception as e:
                    logger.error(f"File {i+1} failed: {e}")
                    results.append(
                        {
                            "partition_id": i,
                            "input_path": file_path,
                            "output_path": None,
                            "success": False,
                            "error": str(e),
                        }
                    )

        # Sort results by partition_id to maintain order
        results.sort(key=lambda x: x["partition_id"])

        # Log results
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        logger.info(f"Processing completed: {successful} successful, {failed} failed")

        return results

    def _load_raw_partitions(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load raw partitions without processing (for when there are no pre-convergence operations).
        """
        logger.info("Loading raw partitions...")
        results = []

        for i, file_path in enumerate(file_paths):
            try:
                # Create DatasetBuilder instance for this partition
                from data_juicer.core.data.dataset_builder import DatasetBuilder

                partition_cfg = Namespace(**vars(self.cfg))
                partition_cfg.dataset_path = file_path
                datasetbuilder = DatasetBuilder(partition_cfg, executor_type="ray")

                # Load dataset
                dataset = datasetbuilder.load_dataset()

                results.append(
                    {
                        "partition_id": i,
                        "input_path": file_path,
                        "dataset": dataset,  # Return dataset object instead of file path
                        "success": True,
                        "error": None,
                    }
                )

                logger.info(f"Loaded raw partition {i}")

            except Exception as e:
                logger.error(f"Failed to load raw partition {i}: {e}")
                results.append(
                    {"partition_id": i, "input_path": file_path, "output_path": None, "success": False, "error": str(e)}
                )

        return results

    def _merge_partitions_for_global_ops(self, partition_results: List[Dict[str, Any]]):
        """
        Merge processed partitions into a single dataset for global operations.

        Uses Ray's efficient union-based merging for all dataset sizes.
        """
        logger.info("Merging partitions for global operations...")

        # Filter successful results
        successful_results = [r for r in partition_results if r["success"]]

        if not successful_results:
            raise RuntimeError("No successful partitions to merge")

        return self._union_merge_partitions(successful_results)

    def _union_merge_partitions(self, successful_results: List[Dict[str, Any]]):
        """
        Merge partitions using Ray's union operation.

        This approach unions all datasets in a single operation, which is more efficient
        than chaining individual unions and reduces object store pressure.
        """
        logger.info("Merging partitions using Ray union operation...")

        # Extract dataset objects directly from results
        partition_datasets = []

        for result in successful_results:
            if "dataset" in result and result["dataset"] is not None:
                partition_datasets.append(result["dataset"].data)

        if not partition_datasets:
            raise RuntimeError("No valid datasets found")

        logger.info(f"Union merge of {len(partition_datasets)} partitions...")

        # Union all datasets in a single operation (more efficient than chaining)
        if len(partition_datasets) == 1:
            merged_dataset = partition_datasets[0]
        else:
            # Use * to unpack the list as individual arguments
            merged_dataset = partition_datasets[0].union(*partition_datasets[1:])

        logger.info(f"Union dataset created")

        # materialize to avoid union chain and memory stress
        return RayDataset(merged_dataset.materialize(), dataset_path="merged", cfg=self.cfg)

    def _process_merged_dataset(self, merged_dataset, post_convergence_ops: List):
        """
        Process the merged dataset with post-convergence operations (global operations).
        """
        logger.info(f"Processing merged dataset with {len(post_convergence_ops)} global operations...")

        # Apply post-convergence operations
        processed_dataset = merged_dataset.process(post_convergence_ops)

        logger.info(f"Global operations completed. Final dataset ready for export")

        return processed_dataset

    def _load_processed_dataset_from_results(self, partition_results: List[Dict[str, Any]]):
        """
        Load processed dataset from partition results using Ray's union operation.

        Args:
            partition_results: List of results from parallel processing

        Returns:
            Combined Ray dataset
        """
        logger.info("Loading processed dataset from partition results...")

        # Filter successful results
        successful_results = [r for r in partition_results if r["success"]]
        failed_results = [r for r in partition_results if not r["success"]]

        if failed_results:
            logger.warning(f"{len(failed_results)} files failed processing:")
            for result in failed_results:
                logger.warning(f"  File {result['partition_id']}: {result['error']}")

        if not successful_results:
            raise RuntimeError("All files failed processing")

        # Extract dataset objects directly from results
        partition_datasets = []
        for result in successful_results:
            if "dataset" in result and result["dataset"] is not None:
                try:
                    partition_datasets.append(result["dataset"].data)
                    logger.info(f"Using partition {result['partition_id']} with {result['sample_count']} samples")

                except Exception as e:
                    logger.error(f"Failed to use partition {result['partition_id']}: {e}")
                    continue

        if not partition_datasets:
            raise RuntimeError("No partition datasets could be loaded successfully")

        # Combine all partitions using Ray's union operation
        logger.info(f"Combining {len(partition_datasets)} partitions...")
        # Use the first dataset and union the rest to avoid chaining
        combined_dataset = partition_datasets[0]
        for dataset in partition_datasets[1:]:
            combined_dataset = combined_dataset.union(dataset)

        # total_samples = combined_dataset.count()
        # logger.info(f"Successfully combined dataset with {total_samples} total samples")

        return combined_dataset

    def _prepare_operators(self):
        """Prepare process operators."""
        ops = load_ops(self.cfg.process)

        if self.cfg.op_fusion:
            probe_res = None
            if self.cfg.fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                probe_res, _ = self.adapter.probe_small_batch(self.dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy [{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        return ops

    def _load_existing_partitions(self):
        """Load existing partitions for resumption."""
        if not self.dataset_mapping:
            return []

        # Use correct file extension based on storage format
        if self.storage_format == "parquet":
            extension = ".parquet"
        elif self.storage_format == "arrow":
            extension = ".arrow"
        else:
            extension = ".jsonl"

        partition_paths = [
            os.path.join(self.cfg.partition_dir, f"partition_{i:06d}{extension}")
            for i in range(self.dataset_mapping.partition_count)
        ]

        return partition_paths

    def _load_partition_result(self, partition_id: int):
        """Load existing result for a completed partition."""
        # This would load the existing result from the partition's output
        # For now, return a basic structure
        return {
            "partition_id": partition_id,
            "input_path": f"partition_{partition_id:06d}.parquet",
            "output_path": f"partition_{partition_id:06d}_processed.parquet",
            "success": True,
            "sample_count": 0,  # This should be loaded from the actual result
        }

    def _load_processed_dataset(self, output_path: str):
        """Load the final processed dataset."""
        # Convert relative path to absolute path to avoid Ray path resolution issues
        output_path = os.path.abspath(output_path)

        # Use the RayExporter's format detection logic
        export_format = self.exporter.export_format

        if export_format == "parquet":
            return ray.data.read_parquet(output_path)
        elif export_format == "arrow":
            return ray.data.read_arrow(output_path)
        elif export_format in ["json", "jsonl"]:
            return ray.data.read_json(output_path)
        else:
            # Fallback to JSONL for unknown formats
            return ray.data.read_json(output_path)

    def _override_strategy_methods(self):
        """Override strategy methods for partitioned execution."""
        # Override partition count determination
        self._determine_partition_count = self._determine_partition_count_partitioned
        self._analyze_dataset_size = self._analyze_dataset_size_partitioned
        self._detect_convergence_points = self._detect_convergence_points_partitioned
        self._get_dag_node_for_operation = self._get_dag_node_for_operation_partitioned

    def _determine_partition_count_partitioned(self, cfg) -> int:
        """Determine partition count for partitioned execution."""
        # If we already have dataset_mapping (from actual partitioning), use that
        if hasattr(self, "dataset_mapping") and self.dataset_mapping:
            logger.info(f"Using actual partition count from dataset mapping: {self.dataset_mapping.partition_count}")
            return self.dataset_mapping.partition_count

        if self.auto_configure_resources:
            # Will be determined after dataset loading
            return 1  # Placeholder
        else:
            # Use configured partition size
            dataset_size = self._analyze_dataset_size_partitioned(cfg.dataset_path)
            estimated_count = max(1, dataset_size // self.partition_size)
            logger.info(
                f"Estimated partition count: {estimated_count} (dataset_size={dataset_size}, partition_size={self.partition_size})"
            )
            return estimated_count

    def _analyze_dataset_size_partitioned(self, dataset_path: str) -> int:
        """Analyze dataset size for partition count determination."""
        try:
            import os

            file_size = os.path.getsize(dataset_path)
            # More accurate estimate for partitioned execution
            estimated_lines = file_size // 512  # Assume 512 bytes per line
            return estimated_lines
        except Exception as e:
            logger.error(f"Error analyzing dataset size: {e}")
            # Fallback to default
            return 100000

    def _detect_convergence_points_partitioned(self, cfg) -> List[int]:
        """Detect convergence points for partitioned execution."""
        operations = self._prepare_operators()
        convergence_points = []

        for op_idx, op in enumerate(operations):
            # Detect global operations (deduplicators, etc.)
            if self._is_global_operation_partitioned(op):
                convergence_points.append(op_idx)

            # Detect manual convergence points
            if hasattr(op, "converge_after") and op.converge_after:
                convergence_points.append(op_idx)

        return convergence_points

    def _is_global_operation_partitioned(self, operation) -> bool:
        """Check if an operation is a global operation for partitioned execution."""
        # Deduplicators are typically global operations
        if hasattr(operation, "_name") and "deduplicator" in operation._name:
            return True

        # Check for explicit global operation flag
        if hasattr(operation, "is_global_operation") and operation.is_global_operation:
            return True

        return False

    def _get_dag_node_for_operation_partitioned(
        self, op_name: str, op_idx: int, partition_id: int, **kwargs
    ) -> Optional[str]:
        """Get DAG node ID for partitioned operation."""
        if not self.dag_execution_strategy:
            return None

        return self.dag_execution_strategy.get_dag_node_id(op_name, op_idx, partition_id=partition_id, **kwargs)
