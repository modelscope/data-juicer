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
import shutil
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
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


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


class PartitionedRayExecutor(ExecutorBase, EventLoggingMixin):
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

        # Partitioning configuration
        # Support both flat and nested partition configuration
        partition_config = getattr(self.cfg, "partition", {})

        # Check if auto-configuration is enabled
        self.auto_configure_partitions = partition_config.get("auto_configure", False)

        if self.auto_configure_partitions:
            logger.info("Auto-configuration enabled - will analyze dataset and optimize partition size")
            # We'll configure this after loading the dataset
            self.partition_size = None
            self.max_partition_size_mb = None
        else:
            # Read from nested partition config first, fall back to flat config
            self.partition_size = partition_config.get("size") or getattr(self.cfg, "partition_size", 10000)
            self.max_partition_size_mb = partition_config.get("max_size_mb") or getattr(
                self.cfg, "max_partition_size_mb", 128
            )

        # Fault tolerance configuration (now under partition section)
        partition_config = getattr(self.cfg, "partition", {})
        self.enable_fault_tolerance = partition_config.get("enable_fault_tolerance") or getattr(
            self.cfg, "enable_fault_tolerance", True
        )
        self.max_retries = partition_config.get("max_retries") or getattr(self.cfg, "max_retries", 3)
        self.retry_backoff = partition_config.get("retry_backoff", "exponential")

        # Intermediate storage configuration (includes file lifecycle management)
        intermediate_storage_config = getattr(self.cfg, "intermediate_storage", {})
        self.storage_format = intermediate_storage_config.get(
            "format", "parquet"
        )  # parquet, arrow, jsonl - for disk storage
        self.storage_compression = intermediate_storage_config.get("compression", "snappy")
        self.parquet_batch_size = intermediate_storage_config.get(
            "parquet_batch_size", 10000
        )  # Number of rows per parquet file
        logger.info(f"Using parquet batch size: {self.parquet_batch_size} rows per file")

        # File lifecycle management (now part of intermediate_storage config)
        self.preserve_intermediate_data = intermediate_storage_config.get("preserve_intermediate_data") or getattr(
            self.cfg, "preserve_intermediate_data", False
        )
        self.cleanup_temp_files = intermediate_storage_config.get("cleanup_temp_files", True)
        self.cleanup_on_success = intermediate_storage_config.get("cleanup_on_success", False)
        self.retention_policy = intermediate_storage_config.get("retention_policy", "keep_all")
        self.max_retention_days = intermediate_storage_config.get("max_retention_days", 7)

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

        # Initialize Ray
        logger.info("Initializing Ray for partitioned execution...")
        # Suppress macOS malloc stack logging warnings
        os.environ["MALLOC_NANOZONE"] = "0"
        ray.init(getattr(self.cfg, "ray_address", "auto"))
        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # Initialize dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        # Use resolved directory paths from config (already handled by config.py)
        self.partitions_dir = self.cfg.partition_dir
        self.intermediate_dir = self.cfg.intermediate_dir
        self.checkpoint_dir = self.cfg.checkpoint_dir
        self.results_dir = self.cfg.results_dir
        self.metadata_dir = self.cfg.metadata_dir
        self.logs_dir = self.cfg.event_log_dir
        self.events_file = self.cfg.event_log_file
        self.summary_file = os.path.join(self.logs_dir, "processing_summary.json")

        # Create directories (already created by config.py, but ensure they exist)
        for dir_path in [
            self.partitions_dir,
            self.intermediate_dir,
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
            # Use configurable batch size for optimal file sizes
            ray_dataset.write_parquet(
                abs_file_path, num_rows_per_file=self.parquet_batch_size, compression=self.storage_compression
            )
        elif format_type == "arrow":
            # Convert to pandas and then to Arrow format
            import pyarrow as pa
            import pyarrow.feather as feather

            # Convert to pandas DataFrame first, then to Arrow table
            df = ray_dataset.to_pandas()
            table = pa.Table.from_pandas(df)

            # Write as Feather format
            feather.write_feather(table, abs_file_path)
        else:  # jsonl
            ray_dataset.write_json(abs_file_path, force_ascii=False)

    def _estimate_partition_count(self, dataset) -> int:
        """Estimate the number of partitions based on dataset size."""
        try:
            total_samples = dataset.data.count()
            # Use ceiling division to ensure we have enough partitions for all data
            # Formula: (total_samples + partition_size - 1) // partition_size
            # This ensures that partial partitions are included
            # Example: 356317 samples with 50000 partition size = 8 partitions
            # (356317 + 50000 - 1) // 50000 = 406316 // 50000 = 8
            return max(1, (total_samples + self.partition_size - 1) // self.partition_size)
        except Exception:
            # Fallback to file-based estimation
            return max(1, int(ray.cluster_resources().get("CPU", 1) * 2))

    def _create_partitions_with_mapping(self, dataset) -> Tuple[List[str], DatasetMapping]:
        """Create partitions from the dataset with preserved mapping."""
        logger.info("Creating dataset partitions with mapping...")

        # Get original dataset information
        original_dataset_path = self.cfg.dataset_path
        total_samples = dataset.data.count()

        # Estimate partition count
        partition_count = self._estimate_partition_count(dataset)
        logger.info(f"Creating {partition_count} partitions from {total_samples} samples...")

        # Create partitions using Ray's repartition
        partitioned_dataset = dataset.data.repartition(partition_count)

        # Initialize dataset mapping
        self.dataset_mapping = DatasetMapping(
            original_dataset_path=original_dataset_path,
            original_dataset_size=total_samples,
            partition_count=partition_count,
            partition_size=self.partition_size,
        )

        # Save partitions to disk with metadata
        partition_paths = []

        # Get partitions from Ray's repartitioned dataset
        # Use get_internal_block_refs() to get the actual partitions
        block_refs = partitioned_dataset.get_internal_block_refs()

        for i, block_ref in enumerate(block_refs):
            # Start with base path, will be updated based on storage format
            # Add .parquet suffix for consistency with other parquet directories
            partition_path = os.path.join(self.partitions_dir, f"partition_{i:06d}.parquet")

            # Get the actual partition data from the block reference
            partition_data = ray.get(block_ref)

            # Debug: log the type and structure of partition_data
            logger.debug(f"Partition {i}: partition_data type = {type(partition_data)}")
            if hasattr(partition_data, "shape"):
                logger.debug(f"Partition {i}: partition_data shape = {partition_data.shape}")
            elif hasattr(partition_data, "num_rows"):
                logger.debug(f"Partition {i}: partition_data num_rows = {partition_data.num_rows}")
            elif isinstance(partition_data, list):
                logger.debug(f"Partition {i}: partition_data length = {len(partition_data)}")
                if partition_data:
                    logger.debug(f"Partition {i}: first item type = {type(partition_data[0])}")
                    logger.debug(f"Partition {i}: first item = {partition_data[0]}")

            # Convert PyArrow table to list of dictionaries for processing
            if hasattr(partition_data, "to_pandas"):
                # If it's a PyArrow table, convert to pandas then to dict
                df = partition_data.to_pandas()
                partition_data = df.to_dict("records")
                # Validate that we have proper dictionaries
                if partition_data and not isinstance(partition_data[0], dict):
                    logger.error(f"Invalid data structure: expected dict, got {type(partition_data[0])}")
                    # Try alternative conversion
                    partition_data = [row.to_dict() for _, row in df.iterrows()]
            elif isinstance(partition_data, list):
                # If it's already a list, validate it contains dictionaries
                if partition_data and not isinstance(partition_data[0], dict):
                    logger.error(f"Invalid data structure in list: expected dict, got {type(partition_data[0])}")
                    # Try to convert to proper format
                    partition_data = [
                        item if isinstance(item, dict) else {"text": str(item)} for item in partition_data
                    ]
            else:
                # Fallback: try to convert to list
                partition_data = list(partition_data)
                # Validate the converted data
                if partition_data and not isinstance(partition_data[0], dict):
                    logger.error(
                        f"Invalid data structure after list conversion: expected dict, got {type(partition_data[0])}"
                    )
                    partition_data = [{"text": str(item)} for item in partition_data]

            # Calculate metadata
            sample_count = len(partition_data)
            checksum = self._calculate_checksum(partition_data)

            # Calculate approximate start/end indices for metadata
            # Since we're using Ray's repartition, we can only approximate
            start_idx = i * (total_samples // partition_count)
            end_idx = min(start_idx + sample_count, total_samples)

            # Save partition to disk using configurable format
            if self.storage_format == "parquet":
                # Use Parquet for best performance and compression
                # Use absolute path to avoid Ray worker file system issues
                partition_path_abs = os.path.abspath(partition_path)
                os.makedirs(partition_path_abs, exist_ok=True)
                partition_dataset = ray.data.from_items(partition_data)
                # Use configurable batch size for optimal file sizes
                partition_dataset.write_parquet(
                    partition_path_abs, num_rows_per_file=self.parquet_batch_size, compression=self.storage_compression
                )
                partition_path = partition_path_abs
            elif self.storage_format == "arrow":
                # Use Arrow (Feather) for memory mapping and zero-copy reads
                # Append .arrow extension for proper file identification
                partition_path = partition_path + ".arrow"
                # Ensure directory exists before writing
                os.makedirs(os.path.dirname(partition_path), exist_ok=True)

                # Convert to pandas and then to Arrow format
                import pyarrow as pa
                import pyarrow.feather as feather

                partition_dataset = ray.data.from_items(partition_data)
                df = partition_dataset.to_pandas()
                table = pa.Table.from_pandas(df)
                feather.write_feather(table, partition_path)
            else:
                # Fallback to JSONL for compatibility
                partition_path = partition_path + ".jsonl"
                with open(partition_path, "w") as f:
                    for sample in partition_data:
                        f.write(json.dumps(sample) + "\n")

            # Get file size
            file_size = os.path.getsize(partition_path)

            # Create partition metadata
            partition_metadata = PartitionMetadata(
                partition_id=i,
                original_start_idx=start_idx,
                original_end_idx=end_idx,
                sample_count=sample_count,
                file_size_bytes=file_size,
                checksum=checksum,
                created_timestamp=time.time(),
            )

            if self.dataset_mapping.partitions is not None:
                self.dataset_mapping.partitions.append(partition_metadata)
            partition_paths.append(partition_path)

            logger.info(
                f"Created partition {i+1}/{partition_count}: {partition_path} "
                f"({sample_count} samples, {file_size} bytes)"
            )

        # Save dataset mapping
        self._save_dataset_mapping()

        return partition_paths, self.dataset_mapping

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
        partition_intermediate_dir = os.path.join(self.intermediate_dir, f"partition_{partition_id:06d}")

        if not os.path.exists(partition_intermediate_dir):
            return None, None

        # Find all operation checkpoint directories (Ray creates directories, not files)
        checkpoint_dirs = []
        for item in os.listdir(partition_intermediate_dir):
            item_path = os.path.join(partition_intermediate_dir, item)
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
        checkpoint_path = os.path.join(partition_intermediate_dir, latest_dir)

        logger.info(f"Found operation checkpoint for partition {partition_id}: op_{latest_op_idx} at {checkpoint_path}")
        return latest_op_idx, checkpoint_path

    def _load_operation_checkpoint(self, checkpoint_path: str) -> ray.data.Dataset:
        """Load dataset from operation checkpoint."""
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
        partition_intermediate_dir = None

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
                # Wrap in RayDataset for processing
                from data_juicer.core.data.ray_dataset import RayDataset

                current_dataset = RayDataset(raw_dataset, dataset_path=partition_path, cfg=self.cfg)
                ops_to_process = ops[latest_op_idx + 1 :]
            else:
                # Load partition data
                logger.debug(f"Loading partition {partition_id} from {partition_path}")
                if partition_path.endswith(".parquet"):
                    raw_dataset = ray.data.read_parquet(partition_path)
                elif partition_path.endswith(".arrow"):
                    raw_dataset = ray.data.read_arrow(partition_path)
                else:
                    raw_dataset = ray.data.read_json(partition_path)
                # Wrap in RayDataset for processing
                from data_juicer.core.data.ray_dataset import RayDataset

                current_dataset = RayDataset(raw_dataset, dataset_path=partition_path, cfg=self.cfg)
                ops_to_process = ops

            # Create intermediate directory if preserving intermediate data
            if self.preserve_intermediate_data:
                partition_intermediate_dir = os.path.join(self.intermediate_dir, f"partition_{partition_id:06d}")
                os.makedirs(partition_intermediate_dir, exist_ok=True)

            # Process operations using RayDataset.process method
            op_start_time = time.time()
            input_rows = current_dataset.data.count()

            # Note: Ray tasks are automatically terminated when the main process is killed
            # No need to track individual Ray job IDs

            # Apply all operations at once using RayDataset.process
            current_dataset.process(ops_to_process)

            op_duration = time.time() - op_start_time
            output_rows = current_dataset.data.count()

            # Log operation completion for all operations
            for op_idx, op in enumerate(ops_to_process):
                actual_op_idx = latest_op_idx + 1 + op_idx if latest_op_idx is not None else op_idx

                # Log operation start
                self.log_op_start(partition_id, op.__class__.__name__, actual_op_idx, {})

                # Determine checkpoint path
                checkpoint_path = None
                if self._should_checkpoint(actual_op_idx, op.__class__.__name__, partition_id):
                    if self.preserve_intermediate_data:
                        checkpoint_path = os.path.join(
                            partition_intermediate_dir, f"op_{actual_op_idx:03d}_{op.__class__.__name__}.parquet"
                        )
                        self._write_dataset_with_directory_creation(current_dataset, checkpoint_path, "parquet")

                        # Log checkpoint save
                        self.log_checkpoint_save(partition_id, op.__class__.__name__, actual_op_idx, checkpoint_path)
                        logger.debug(f"Saved checkpoint for partition {partition_id}, operation {actual_op_idx}")

                # Log operation completion
                self.log_op_complete(
                    partition_id,
                    op.__class__.__name__,
                    actual_op_idx,
                    op_duration,
                    checkpoint_path,
                    input_rows,
                    output_rows,
                )

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
                "intermediate_dir": partition_intermediate_dir,
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

                    # Calculate backoff delay based on strategy
                    if self.retry_backoff == "exponential":
                        delay = 2**attempt
                    elif self.retry_backoff == "linear":
                        delay = attempt + 1
                    else:  # fixed
                        delay = 1

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
                        "intermediate_dir": None,
                        "success": False,
                        "error": str(e),
                        "retry_count": attempt + 1,
                    }

    def _merge_partitions_with_mapping(self, partition_results: List[Dict[str, Any]]) -> str:
        """Merge processed partitions into final output with mapping preservation."""
        logger.info("Merging processed partitions...")

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

        # Merge successful partitions
        # Ensure the export path is treated as a file, not a directory
        export_path = self.cfg.export_path
        if os.path.isdir(export_path):
            # If it's a directory, create a file inside it
            export_path = os.path.join(export_path, "processed.jsonl")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        def convert_datetime_to_str(obj):
            """Recursively convert datetime objects to ISO format strings."""
            if hasattr(obj, "isoformat"):  # Handle datetime objects
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: convert_datetime_to_str(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_str(item) for item in obj]
            else:
                return obj

        with open(export_path, "w") as output_file:
            for result in successful_results:
                if result["output_path"] and os.path.exists(result["output_path"]):
                    # Handle different file formats
                    if result["output_path"].endswith(".parquet"):
                        # For parquet files, we need to read and convert to JSONL
                        import pandas as pd

                        df = pd.read_parquet(result["output_path"])
                        for _, row in df.iterrows():
                            # Convert datetime objects to strings for JSON serialization
                            row_dict = convert_datetime_to_str(row.to_dict())
                            output_file.write(json.dumps(row_dict) + "\n")
                    elif result["output_path"].endswith(".arrow"):
                        import pyarrow.feather as feather

                        table = feather.read_feather(result["output_path"])
                        # Handle both PyArrow Table and pandas DataFrame
                        if hasattr(table, "to_pandas"):
                            # PyArrow Table - convert to pandas
                            df = table.to_pandas()
                        else:
                            # Already a pandas DataFrame
                            df = table

                        for _, row in df.iterrows():
                            # Convert datetime objects to strings for JSON serialization
                            row_dict = convert_datetime_to_str(row.to_dict())
                            output_file.write(json.dumps(row_dict) + "\n")
                    else:
                        # For JSONL files, copy directly
                        with open(result["output_path"], "r") as input_file:
                            shutil.copyfileobj(input_file, output_file)

        # Create final mapping report
        self._create_final_mapping_report(partition_results)

        logger.info(f"Merged {len(successful_results)} partitions into {export_path}")
        return export_path

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
        if hasattr(self.cfg, "job_id") and self.cfg.job_id:
            logger.info(f"🔍 Checking for job resumption: {self.cfg.job_id}")

            # Validate resumption using event analysis
            resumption_validation = self._validate_job_resumption(self.cfg.job_id)

            if resumption_validation["can_resume"]:
                logger.info(f"🔄 Resuming job: {self.cfg.job_id}")
                return self._resume_job(resumption_validation)
            else:
                logger.error(f"❌ Job resumption failed: {resumption_validation.get('reason', 'Unknown')}")
                logger.error(f"   Cannot resume job {self.cfg.job_id} with the same job_id")
                logger.error(f"   Please use a different job_id or fix the validation issues")
                raise RuntimeError(f"Job resumption failed: {resumption_validation.get('reason', 'Unknown')}")

        # Create job summary at the start of the run
        self._create_job_summary(self.cfg.job_id, self.cfg.job_dir)

        # Load dataset
        logger.info("Loading dataset with Ray...")
        dataset = self._load_dataset(load_data_np)

        # Prepare process operators
        logger.info("Preparing process operators...")
        ops = self._prepare_operators()

        # Create partitions
        logger.info("Creating new partitions...")
        partition_paths, self.dataset_mapping = self._create_partitions_with_mapping(dataset)

        # Log job start event
        job_config = {
            "dataset_path": self.cfg.dataset_path,
            "work_dir": self.cfg.work_dir,
            "executor_type": self.cfg.executor_type,
            "partition_size": getattr(self.cfg, "partition_size", None),
            "max_partition_size_mb": getattr(self.cfg, "max_partition_size_mb", None),
            "checkpoint_enabled": (
                getattr(self.cfg.checkpoint, "enabled", False) if hasattr(self.cfg, "checkpoint") else False
            ),
            "checkpoint_strategy": (
                getattr(self.cfg.checkpoint, "strategy", None) if hasattr(self.cfg, "checkpoint") else None
            ),
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
            "enable_fault_tolerance": (
                getattr(self.cfg.partition, "enable_fault_tolerance", True) if hasattr(self.cfg, "partition") else True
            ),
            "max_retries": getattr(self.cfg.partition, "max_retries", 3) if hasattr(self.cfg, "partition") else 3,
            "retry_backoff": (
                getattr(self.cfg.partition, "retry_backoff", "exponential")
                if hasattr(self.cfg, "partition")
                else "exponential"
            ),
        }

        self.log_job_start(job_config, len(partition_paths))

        # Process partitions with fault tolerance in parallel using threads
        logger.info(f"Processing {len(partition_paths)} partitions with fault tolerance in parallel using threads...")

        # Use ThreadPoolExecutor for parallel processing
        partition_results = []
        with ThreadPoolExecutor(max_workers=min(len(partition_paths), 8)) as executor:
            # Submit all partition processing tasks
            future_to_partition = {}
            for i, partition_path in enumerate(partition_paths):
                logger.info(f"Submitting partition {i}/{len(partition_paths)-1} for parallel processing")
                future = executor.submit(self._process_partition_with_retry, partition_path, ops, i)
                future_to_partition[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_partition):
                partition_id = future_to_partition[future]
                try:
                    result = future.result()
                    partition_results.append(result)
                    logger.info(f"Partition {partition_id} completed successfully")
                except Exception as e:
                    logger.error(f"Partition {partition_id} failed with exception: {e}")
                    partition_results.append(
                        {
                            "partition_id": partition_id,
                            "input_path": partition_paths[partition_id],
                            "output_path": None,
                            "intermediate_dir": None,
                            "success": False,
                            "error": str(e),
                        }
                    )

        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        self._save_checkpoint(partition_results, ops)

        # Merge partitions
        logger.info("Merging processed partitions...")
        final_output_path = self._merge_partitions_with_mapping(partition_results)

        # Log job completion
        job_duration = time.time() - job_start_time
        self.log_job_complete(job_duration, final_output_path)

        logger.info(f"✅ Job completed successfully in {job_duration:.2f}s")
        logger.info(f"📁 Output saved to: {final_output_path}")

        if skip_return:
            return None

        # Return processed dataset
        return self._load_processed_dataset(final_output_path)

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
                                "intermediate_dir": None,
                                "success": False,
                                "error": str(e),
                            }
                        )

        # Save final checkpoint
        logger.info("Saving final checkpoint...")
        self._save_checkpoint(partition_results, ops)

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
        return self.datasetbuilder.load_dataset(num_proc=load_data_np)

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
        if self.cfg.storage_format == "parquet":
            extension = ".parquet"
        elif self.cfg.storage_format == "arrow":
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
        if self.cfg.storage_format == "parquet":
            return ray.data.read_parquet(output_path)
        elif self.cfg.storage_format == "arrow":
            return ray.data.read_arrow(output_path)
        else:
            return ray.data.read_json(output_path)
