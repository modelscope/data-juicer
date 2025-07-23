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


class PartitionedRayExecutor(ExecutorBase):
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

        # Partitioning configuration
        self.partition_size = getattr(self.cfg, "partition_size", 10000)  # samples per partition
        self.max_partition_size_mb = getattr(self.cfg, "max_partition_size_mb", 128)
        self.enable_fault_tolerance = getattr(self.cfg, "enable_fault_tolerance", True)
        self.max_retries = getattr(self.cfg, "max_retries", 3)
        self.preserve_intermediate_data = getattr(self.cfg, "preserve_intermediate_data", False)

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

        # Data format configuration for performance
        self.storage_format = getattr(self.cfg, "storage_format", "parquet")  # parquet, arrow, jsonl - for disk storage
        self.use_arrow_batches = getattr(
            self.cfg, "use_arrow_batches", True
        )  # Use Arrow batch format for processing (recommended)
        self.arrow_batch_size = getattr(self.cfg, "arrow_batch_size", 1000)  # Arrow batch size for processing

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

    def _log_event(self, event: ProcessingEvent):
        """Log a processing event."""
        # Write to JSONL file
        with open(self.events_file, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

        # Update summary
        if event.event_type == "partition_start":
            self.event_summary["total_partitions"] += 1
        elif event.event_type == "partition_complete":
            self.event_summary["completed_partitions"] += 1
        elif event.event_type == "partition_failed":
            self.event_summary["failed_partitions"] += 1
        elif event.event_type == "operation_checkpoint":
            self.event_summary["checkpoints_created"] += 1
        elif event.event_type == "error":
            self.event_summary["errors"].append(
                {
                    "timestamp": event.timestamp,
                    "message": event.message,
                    "partition_id": event.partition_id,
                    "operation_name": event.operation_name,
                }
            )

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
        for i in range(partition_count):
            partition_path = os.path.join(self.partitions_dir, f"partition_{i:06d}.jsonl")

            # Get partition data
            partition_data = list(partitioned_dataset.take_partition(i))

            # Calculate metadata
            sample_count = len(partition_data)
            start_idx = i * self.partition_size
            end_idx = min(start_idx + sample_count, total_samples)
            checksum = self._calculate_checksum(partition_data)

            # Save partition to disk using configurable format
            if self.storage_format == "parquet":
                # Use Parquet for best performance and compression
                partition_path = partition_path.replace(".jsonl", ".parquet")
                partition_dataset = ray.data.from_items(partition_data)
                partition_dataset.write_parquet(partition_path)
            elif self.storage_format == "arrow":
                # Use Arrow (Feather) for memory mapping and zero-copy reads
                partition_path = partition_path.replace(".jsonl", ".arrow")
                partition_dataset = ray.data.from_items(partition_data)
                # Convert to Arrow table and save as Feather format
                import pyarrow as pa
                import pyarrow.feather as feather

                df = partition_dataset.to_pandas()
                table = pa.Table.from_pandas(df)
                feather.write_feather(table, partition_path)
            else:
                # Fallback to JSONL for compatibility
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
            current_dataset = partition_dataset

            for op_idx, op in enumerate(ops):
                logger.debug(f"Applying op {op_idx+1}/{len(ops)}: {op._name} to partition {partition_id}")

                # Save intermediate state if enabled (using configurable format)
                if self._should_checkpoint(op_idx, op._name, partition_id):
                    if self.storage_format == "parquet":
                        intermediate_path = os.path.join(
                            partition_intermediate_dir, f"after_op_{op_idx:03d}_{op._name}.parquet"
                        )
                        current_dataset.write_parquet(intermediate_path)
                    elif self.storage_format == "arrow":
                        intermediate_path = os.path.join(
                            partition_intermediate_dir, f"after_op_{op_idx:03d}_{op._name}.arrow"
                        )
                        # Convert to Arrow table and save as Feather format with compression
                        import pyarrow.feather as feather

                        # Use Arrow batch format for better performance
                        if hasattr(current_dataset, "to_arrow_refs"):
                            # Use Arrow batch format directly if available
                            arrow_refs = current_dataset.to_arrow_refs()
                            tables = ray.get(arrow_refs)
                            if tables:
                                table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
                            else:
                                # Fallback to pandas conversion
                                df = current_dataset.to_pandas()
                                table = pa.Table.from_pandas(df)
                        else:
                            # Fallback to pandas conversion
                            df = current_dataset.to_pandas()
                            table = pa.Table.from_pandas(df)

                        # Save with compression for better storage efficiency
                        feather.write_feather(table, intermediate_path, compression="lz4")
                    else:
                        intermediate_path = os.path.join(
                            partition_intermediate_dir, f"after_op_{op_idx:03d}_{op._name}.jsonl"
                        )
                        current_dataset.write_json(intermediate_path, force_ascii=False)
                    logger.debug(f"Saved intermediate state to {intermediate_path}")

                    # Log checkpoint event for intermediate state
                    self._log_event(
                        ProcessingEvent(
                            event_id=f"op_checkpoint_{partition_id}_{op_idx}_{int(time.time())}",
                            event_type="operation_checkpoint",
                            timestamp=time.time(),
                            partition_id=partition_id,
                            operation_name=op._name,
                            operation_idx=op_idx,
                            message=f"Created checkpoint for {op._name} on partition {partition_id}",
                            metadata={"checkpoint_path": intermediate_path},
                        )
                    )

                # Apply operation
                if hasattr(op, "compute_stats_batched"):
                    current_dataset = op.compute_stats_batched(current_dataset)

                if hasattr(op, "process_batched"):
                    result = list(op.process_batched(current_dataset))
                    # Filter based on result
                    if result and isinstance(result[0], bool):
                        current_dataset = current_dataset.filter(lambda x, i: result[i])

            # Save final processed partition using configurable format
            if self.storage_format == "parquet":
                output_path = os.path.join(self.results_dir, f"partition_{partition_id:06d}_processed.parquet")
                current_dataset.write_parquet(output_path)
            elif self.storage_format == "arrow":
                output_path = os.path.join(self.results_dir, f"partition_{partition_id:06d}_processed.arrow")
                # Convert to Arrow table and save as Feather format with compression
                import pyarrow as pa
                import pyarrow.feather as feather

                # Use Arrow batch format for better performance
                if hasattr(current_dataset, "to_arrow_refs"):
                    # Use Arrow batch format directly if available
                    arrow_refs = current_dataset.to_arrow_refs()
                    tables = ray.get(arrow_refs)
                    if tables:
                        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
                    else:
                        # Fallback to pandas conversion
                        df = current_dataset.to_pandas()
                        table = pa.Table.from_pandas(df)
                else:
                    # Fallback to pandas conversion
                    df = current_dataset.to_pandas()
                    table = pa.Table.from_pandas(df)

                # Save with compression for better storage efficiency
                feather.write_feather(table, output_path, compression="lz4")
            else:
                output_path = os.path.join(self.results_dir, f"partition_{partition_id:06d}_processed.jsonl")
                current_dataset.write_json(output_path, force_ascii=False)

            # Create checkpoint at partition completion if strategy is "every_partition"
            if self.checkpoint_strategy == CheckpointStrategy.EVERY_PARTITION and self.checkpoint_enabled:
                partition_checkpoint_path = os.path.join(
                    self.checkpoint_dir, f"partition_{partition_id:06d}_final.parquet"
                )
                current_dataset.write_parquet(partition_checkpoint_path)

                # Log checkpoint event
                self._log_event(
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
                    time.sleep(2**attempt)  # Exponential backoff
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
        with open(self.cfg.export_path, "w") as output_file:
            for result in successful_results:
                if result["output_path"] and os.path.exists(result["output_path"]):
                    with open(result["output_path"], "r") as input_file:
                        shutil.copyfileobj(input_file, output_file)

        # Create final mapping report
        self._create_final_mapping_report(partition_results)

        logger.info(f"Merged {len(successful_results)} partitions into {self.cfg.export_path}")
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
            return json.load(f)

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

        # 3. Check for existing checkpoint and dataset mapping
        checkpoint_data = None
        if self.enable_fault_tolerance:
            checkpoint_data = self._load_checkpoint()
            if checkpoint_data:
                logger.info("Found existing checkpoint, resuming from previous state...")
                # Restore dataset mapping from checkpoint
                if checkpoint_data.get("dataset_mapping"):
                    self.dataset_mapping = DatasetMapping(**checkpoint_data["dataset_mapping"])

        # 4. Create partitions or load existing mapping
        if checkpoint_data and self.dataset_mapping:
            # Resume from checkpoint
            partition_paths = [
                os.path.join(self.partitions_dir, f"partition_{i:06d}.jsonl")
                for i in range(self.dataset_mapping.partition_count)
            ]
            completed_partitions = {r["partition_id"] for r in checkpoint_data["partition_results"] if r["success"]}
        else:
            # Load or create dataset mapping
            self.dataset_mapping = self._load_dataset_mapping()
            if self.dataset_mapping:
                logger.info("Found existing dataset mapping, using existing partitions...")
                partition_paths = [
                    os.path.join(self.partitions_dir, f"partition_{i:06d}.jsonl")
                    for i in range(self.dataset_mapping.partition_count)
                ]
                completed_partitions = set()
            else:
                # Create new partitions
                partition_paths, self.dataset_mapping = self._create_partitions_with_mapping(dataset)
                completed_partitions = set()

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

        # 8. Cleanup temporary files (if not preserving intermediate data)
        if getattr(self.cfg, "cleanup_temp_files", True) and not self.preserve_intermediate_data:
            logger.info("Cleaning up temporary files...")
            shutil.rmtree(self.partitions_dir, ignore_errors=True)
            shutil.rmtree(self.results_dir, ignore_errors=True)
            if not self.preserve_intermediate_data:
                shutil.rmtree(self.intermediate_dir, ignore_errors=True)

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
