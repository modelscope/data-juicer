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
from typing import Any, Dict, List, Optional, Tuple

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.adapter import Adapter
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin
from data_juicer.core.executor.partition_size_optimizer import (
    auto_configure_partition_size,
)
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
        self.storage_format = intermediate_storage_config.get("format") or getattr(
            self.cfg, "storage_format", "parquet"
        )  # parquet, arrow, jsonl - for disk storage
        self.storage_compression = intermediate_storage_config.get("compression", "snappy")
        self.use_arrow_batches = intermediate_storage_config.get("use_arrow_batches") or getattr(
            self.cfg, "use_arrow_batches", True
        )  # Use Arrow batch format for processing (recommended)
        self.arrow_batch_size = intermediate_storage_config.get("arrow_batch_size") or getattr(
            self.cfg, "arrow_batch_size", 1000
        )  # Arrow batch size for processing
        self.arrow_memory_mapping = intermediate_storage_config.get("arrow_memory_mapping") or getattr(
            self.cfg, "arrow_memory_mapping", False
        )

        # File lifecycle management (now part of intermediate_storage config)
        self.preserve_intermediate_data = intermediate_storage_config.get("preserve_intermediate_data") or getattr(
            self.cfg, "preserve_intermediate_data", False
        )
        self.cleanup_temp_files = intermediate_storage_config.get("cleanup_temp_files", True)
        self.cleanup_on_success = intermediate_storage_config.get("cleanup_on_success", False)
        self.retention_policy = intermediate_storage_config.get("retention_policy", "keep_all")
        self.max_retention_days = intermediate_storage_config.get("max_retention_days", 7)

        # Checkpoint configuration
        checkpoint_cfg = getattr(self.cfg, "checkpoint", {})
        self.checkpoint_enabled = checkpoint_cfg.get("enabled", True)

        # Parse checkpoint strategy with validation
        strategy_str = checkpoint_cfg.get("strategy", "every_op")
        try:
            self.checkpoint_strategy = CheckpointStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Unknown checkpoint strategy: {strategy_str}, defaulting to EVERY_OP")
            self.checkpoint_strategy = CheckpointStrategy.EVERY_OP

        # If strategy is DISABLED, disable checkpointing regardless of enabled flag
        if self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            self.checkpoint_enabled = False

        self.checkpoint_n_ops = checkpoint_cfg.get("n_ops", 1)
        self.checkpoint_op_names = checkpoint_cfg.get("op_names", [])

        # Initialize Ray
        logger.info("Initializing Ray for partitioned execution...")
        ray.init(getattr(self.cfg, "ray_address", "auto"))
        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # Initialize dataset builder
        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        # Partition management directories
        self.partitions_dir = os.path.join(self.work_dir, "partitions")
        self.intermediate_dir = os.path.join(self.work_dir, "intermediate")
        self.checkpoint_dir = os.path.join(self.work_dir, "checkpoints")
        self.results_dir = os.path.join(self.work_dir, "results")
        self.metadata_dir = os.path.join(self.work_dir, "metadata")

        # Create directories
        for dir_path in [
            self.partitions_dir,
            self.intermediate_dir,
            self.checkpoint_dir,
            self.results_dir,
            self.metadata_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # Event logging directory
        self.logs_dir = os.path.join(self.work_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        self.events_file = os.path.join(self.logs_dir, "processing_events.jsonl")
        self.summary_file = os.path.join(self.logs_dir, "processing_summary.json")

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

    def _get_job_specific_paths(self):
        """Get job-specific directory paths if event logging is enabled."""
        if hasattr(self, "event_logger") and self.event_logger:
            # Use job-specific directories
            job_id = self.event_logger.job_id
            job_dir = os.path.join(self.work_dir, job_id)

            # Use configured checkpoint directory if available
            checkpoint_dir = getattr(self.cfg, "checkpoint_dir", None)
            if checkpoint_dir:
                job_checkpoint_dir = os.path.join(checkpoint_dir, job_id)
            else:
                job_checkpoint_dir = os.path.join(job_dir, "checkpoints")

            return {
                "checkpoint_dir": job_checkpoint_dir,
                "metadata_dir": os.path.join(job_dir, "metadata"),
                "partitions_dir": os.path.join(job_dir, "partitions"),
                "intermediate_dir": os.path.join(job_dir, "intermediate"),
                "results_dir": os.path.join(job_dir, "results"),
            }
        else:
            # Use shared directories (backward compatibility)
            return {
                "checkpoint_dir": os.path.join(self.work_dir, "checkpoints"),
                "metadata_dir": os.path.join(self.work_dir, "metadata"),
                "partitions_dir": os.path.join(self.work_dir, "partitions"),
                "intermediate_dir": os.path.join(self.work_dir, "intermediate"),
                "results_dir": os.path.join(self.work_dir, "results"),
            }

    def _update_directories_for_job(self):
        """Update directory paths to use job-specific directories if available."""
        job_paths = self._get_job_specific_paths()
        self.checkpoint_dir = job_paths["checkpoint_dir"]
        self.metadata_dir = job_paths["metadata_dir"]
        self.partitions_dir = job_paths["partitions_dir"]
        self.intermediate_dir = job_paths["intermediate_dir"]
        self.results_dir = job_paths["results_dir"]

        # Create job-specific directories
        for dir_path in [
            self.checkpoint_dir,
            self.metadata_dir,
            self.partitions_dir,
            self.intermediate_dir,
            self.results_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

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
            "success_rate": completed_partitions / total_partitions if total_partitions > 0 else 0,
            "checkpoints_created": self.event_summary["checkpoints_created"],
            "work_directory": self.work_dir,
        }

    def _calculate_checksum(self, data: List[Dict]) -> str:
        """Calculate checksum for partition data."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()

    def _estimate_partition_count(self, dataset) -> int:
        """Estimate the number of partitions based on dataset size."""
        try:
            total_samples = dataset.data.count()
            return max(1, total_samples // self.partition_size)
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
        all_data = list(partitioned_dataset.iter_rows())  # Convert to list for slicing

        for i in range(partition_count):
            # Start with base path, will be updated based on storage format
            partition_path = os.path.join(self.partitions_dir, f"partition_{i:06d}")

            # Get partition data using list slicing
            start_idx = i * self.partition_size
            end_idx = min(start_idx + self.partition_size, total_samples)
            partition_data = all_data[start_idx:end_idx]

            # Calculate metadata
            sample_count = len(partition_data)
            checksum = self._calculate_checksum(partition_data)

            # Save partition to disk using configurable format
            if self.storage_format == "parquet":
                # Use Parquet for best performance and compression
                partition_path = partition_path + ".parquet"
                partition_dataset = ray.data.from_items(partition_data)
                partition_dataset.write_parquet(partition_path)
            elif self.storage_format == "arrow":
                # Use Arrow (Feather) for memory mapping and zero-copy reads
                partition_path = partition_path + ".arrow"
                partition_dataset = ray.data.from_items(partition_data)
                # Convert to Arrow table and save as Feather format
                import pyarrow as pa
                import pyarrow.feather as feather

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

    def _process_partition(self, partition_path: str, ops: List, partition_id: int) -> Dict[str, Any]:
        """Process a single partition with fault tolerance and intermediate data preservation."""
        logger.info(f"Processing partition {partition_id}: {partition_path}")

        # Update partition status
        if (
            self.dataset_mapping
            and self.dataset_mapping.partitions
            and partition_id < len(self.dataset_mapping.partitions)
        ):
            self.dataset_mapping.partitions[partition_id].processing_status = "processing"
            self.dataset_mapping.partitions[partition_id].processing_start_time = time.time()
            self._save_dataset_mapping()

        # Load partition dataset using appropriate format
        if partition_path.endswith(".parquet"):
            partition_dataset = ray.data.read_parquet(partition_path)
        elif partition_path.endswith(".arrow"):
            # Load Arrow (Feather) format with optional memory mapping support
            import pyarrow as pa
            import pyarrow.feather as feather

            # Check if memory mapping is enabled for Arrow files
            use_memory_mapping = getattr(self.cfg, "arrow_memory_mapping", False)

            try:
                if use_memory_mapping:
                    # Use memory mapping for better performance with large files
                    import mmap

                    with open(partition_path, "rb") as f:
                        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                        table = feather.read_table(mm)
                        mm.close()
                    logger.debug(f"Loaded Arrow file with memory mapping: {partition_path}")
                else:
                    # Standard Arrow reading
                    table = feather.read_feather(partition_path)
                    logger.debug(f"Loaded Arrow file: {partition_path}")

                # Validate table before converting to Ray dataset
                if table.num_rows == 0:
                    logger.warning(f"Empty Arrow table loaded from: {partition_path}")

                partition_dataset = ray.data.from_arrow(table)

            except Exception as e:
                logger.error(f"Failed to load Arrow file {partition_path}: {e}")
                # Fallback to standard reading if memory mapping fails
                if use_memory_mapping:
                    logger.info("Falling back to standard Arrow reading")
                    table = feather.read_feather(partition_path)
                    partition_dataset = ray.data.from_arrow(table)
                else:
                    raise
        else:
            partition_dataset = ray.data.read_json(partition_path)

        # Create intermediate data directory for this partition
        partition_intermediate_dir = os.path.join(self.intermediate_dir, f"partition_{partition_id:06d}")
        os.makedirs(partition_intermediate_dir, exist_ok=True)

        # Apply operations with intermediate data preservation
        try:
            # Convert Ray Dataset to pandas for processing to avoid PyArrow schema conflicts
            import pandas as pd

            # Convert to pandas DataFrame
            df = partition_dataset.to_pandas()

            # Create a simple dataset wrapper for processing
            from data_juicer.core.data import NestedDataset

            current_dataset = NestedDataset.from_list(df.to_dict("records"))

            # Process all operations using the standard DataJuicer processing
            current_dataset.process(ops)

            # Get the processed data from the MaterializedDataset
            processed_data = current_dataset.to_list()

            # Convert back to Ray Dataset
            processed_df = pd.DataFrame(processed_data)
            current_dataset = ray.data.from_pandas(processed_df)

            # Save final processed partition using configurable format
            output_path = os.path.join(self.results_dir, f"partition_{partition_id:06d}.{self.storage_format}")

            if self.storage_format == "parquet":
                current_dataset.write_parquet(output_path)
            elif self.storage_format == "arrow":
                # For Arrow format, we need to handle it differently
                if hasattr(current_dataset, "to_arrow_refs"):
                    # Use Arrow references if available
                    _ = current_dataset.to_arrow_refs()
                    # Convert to pandas and then to arrow
                    df = current_dataset.to_pandas()
                else:
                    # Fallback to pandas conversion
                    df = current_dataset.to_pandas()

                table = pa.Table.from_pandas(df)
                with pa.OSFile(output_path, "wb") as sink:
                    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                        writer.write_table(table)
            else:  # jsonl
                current_dataset.write_json(output_path, force_ascii=False)

            # Save partition checkpoint if enabled
            if self.checkpoint_strategy == CheckpointStrategy.EVERY_PARTITION:
                partition_checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"partition_{partition_id:06d}_checkpoint.{self.storage_format}"
                )
                current_dataset.write_parquet(partition_checkpoint_path)

            # Create checkpoint at partition completion if strategy is "every_partition"
            if self.checkpoint_strategy == CheckpointStrategy.EVERY_PARTITION and self.checkpoint_enabled:
                partition_checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"partition_{partition_id:06d}_final.parquet"
                )
                current_dataset.write_parquet(partition_checkpoint_path)

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

            return {
                "partition_id": partition_id,
                "input_path": partition_path,
                "output_path": output_path,
                "intermediate_dir": partition_intermediate_dir,
                "success": True,
                "sample_count": current_dataset.count(),
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

            raise

    def _process_partition_with_retry(self, partition_path: str, ops: List, partition_id: int) -> Dict[str, Any]:
        """Process partition with retry logic for fault tolerance."""
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
                        # For arrow files, convert to JSONL
                        import pyarrow as pa

                        table = pa.ipc.open_file(result["output_path"]).read_all()
                        df = table.to_pandas()
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

    def _save_checkpoint(self, partition_results: List[Dict[str, Any]], ops: List) -> str:
        """Save processing checkpoint with enhanced metadata."""
        checkpoint_data = {
            "timestamp": time.time(),
            "partition_results": partition_results,
            "ops_completed": len(ops),
            "total_partitions": len(partition_results),
            "dataset_mapping": asdict(self.dataset_mapping) if self.dataset_mapping else None,
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{int(time.time())}.json")
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

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

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Run the partitioned dataset processing pipeline.

        Args:
            load_data_np: Number of workers for loading dataset
            skip_return: Whether to skip returning the dataset

        Returns:
            Processed dataset
        """
        # 1. Load dataset
        logger.info("Loading dataset with Ray...")
        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np)

        # Auto-configure partition size if enabled
        if self.auto_configure_partitions:
            logger.info("Running auto-configuration for partition size...")
            try:
                recommendations = auto_configure_partition_size(self.cfg, dataset, self.cfg.process)
                self.partition_size = recommendations["recommended_partition_size"]
                self.max_partition_size_mb = recommendations["recommended_max_size_mb"]
                logger.info(
                    f"Auto-configured partition size: {self.partition_size} samples, {self.max_partition_size_mb} MB max"
                )
            except Exception as e:
                logger.warning(f"Auto-configuration failed: {e}, using default values")
                self.partition_size = 200
                self.max_partition_size_mb = 32

        # Update directories for job-specific paths if event logging is enabled
        self._update_directories_for_job()

        # 2. Extract and prepare operations
        logger.info("Preparing process operators...")
        ops = load_ops(self.cfg.process)

        if self.cfg.op_fusion:
            probe_res = None
            if self.cfg.fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                probe_res, _ = self.adapter.probe_small_batch(dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy [{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        # 3. Check for existing checkpoint
        checkpoint_data = self._load_checkpoint()
        completed_partitions = set()
        if checkpoint_data:
            logger.info("Found existing checkpoint, resuming from previous state...")
            # The checkpoint data already contains reconstructed objects from _load_checkpoint
            self.dataset_mapping = checkpoint_data.get("dataset_mapping")
            completed_partitions = {r["partition_id"] for r in checkpoint_data["partition_results"] if r["success"]}

        # 4. Create partitions or load existing mapping
        if not self.dataset_mapping:
            self.dataset_mapping = self._load_dataset_mapping()

        if self.dataset_mapping:
            logger.info("Found existing dataset mapping, using existing partitions...")
            # Use correct file extension based on storage format
            if self.storage_format == "parquet":
                extension = ".parquet"
            elif self.storage_format == "arrow":
                extension = ".arrow"
            else:
                extension = ".jsonl"

            partition_paths = [
                os.path.join(self.partitions_dir, f"partition_{i:06d}{extension}")
                for i in range(self.dataset_mapping.partition_count)
            ]
        else:
            # Create new partitions
            partition_paths, self.dataset_mapping = self._create_partitions_with_mapping(dataset)

        # 5. Process partitions with fault tolerance
        logger.info(f"Processing {len(partition_paths)} partitions...")
        start_time = time.time()

        partition_results = []
        if checkpoint_data:
            partition_results.extend(checkpoint_data["partition_results"])

        # Process remaining partitions
        remaining_partitions = [(i, path) for i, path in enumerate(partition_paths) if i not in completed_partitions]

        if remaining_partitions:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=ray.cluster_resources().get("CPU", 1)) as executor:
                # Submit partition processing tasks
                future_to_partition = {
                    executor.submit(self._process_partition_with_retry, path, ops, partition_id): (partition_id, path)
                    for partition_id, path in remaining_partitions
                }

                # Collect results
                for future in as_completed(future_to_partition):
                    partition_id, path = future_to_partition[future]
                    try:
                        result = future.result()
                        partition_results.append(result)

                        # Save checkpoint periodically
                        if len(partition_results) % 10 == 0:
                            self._save_checkpoint(partition_results, ops)

                        logger.info(f"Completed partition {partition_id}: {result['success']}")

                    except Exception as e:
                        logger.error(f"Partition {partition_id} failed: {e}")
                        partition_results.append(
                            {
                                "partition_id": partition_id,
                                "input_path": path,
                                "output_path": None,
                                "intermediate_dir": None,
                                "success": False,
                                "error": str(e),
                            }
                        )

        end_time = time.time()
        logger.info(f"Partition processing completed in {end_time - start_time:.2f}s")

        # 6. Merge partitions
        final_output_path = self._merge_partitions_with_mapping(partition_results)

        # 7. Save final checkpoint
        if self.enable_fault_tolerance:
            self._save_checkpoint(partition_results, ops)

        # 8. Cleanup temporary files based on intermediate storage configuration
        if self.cleanup_temp_files:
            if self.retention_policy == "cleanup_all" or (
                self.retention_policy == "keep_failed_only"
                and all(result.get("success", False) for result in partition_results)
            ):
                logger.info("Cleaning up temporary files...")
                shutil.rmtree(self.partitions_dir, ignore_errors=True)
                shutil.rmtree(self.results_dir, ignore_errors=True)
                if not self.preserve_intermediate_data:
                    shutil.rmtree(self.intermediate_dir, ignore_errors=True)
            elif self.retention_policy == "keep_all":
                logger.info("Keeping all intermediate files as per retention policy")
            else:
                logger.info(f"Keeping intermediate files due to retention policy: {self.retention_policy}")

        logger.info(f"Partitioned processing completed. Output: {final_output_path}")

        if not skip_return:
            # Return the processed dataset
            return ray.data.read_json(final_output_path)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and status with mapping information."""
        checkpoint_data = self._load_checkpoint()
        if not checkpoint_data:
            return {"status": "no_checkpoint", "progress": 0}

        total_partitions = checkpoint_data["total_partitions"]
        successful_partitions = len([r for r in checkpoint_data["partition_results"] if r["success"]])
        failed_partitions = total_partitions - successful_partitions

        stats = {
            "status": "completed" if successful_partitions == total_partitions else "in_progress",
            "progress": successful_partitions / total_partitions * 100,
            "total_partitions": total_partitions,
            "successful_partitions": successful_partitions,
            "failed_partitions": failed_partitions,
            "timestamp": checkpoint_data["timestamp"],
        }

        # Add mapping information if available
        if checkpoint_data.get("dataset_mapping"):
            mapping = checkpoint_data["dataset_mapping"]
            stats["original_dataset"] = {
                "path": mapping["original_dataset_path"],
                "size": mapping["original_dataset_size"],
                "partition_size": mapping["partition_size"],
            }

        return stats

    def get_partition_mapping(self) -> Optional[DatasetMapping]:
        """Get the current dataset mapping."""
        return self.dataset_mapping
