#!/usr/bin/env python3
"""
Base class for partitioned executors with comprehensive checkpointing and event logging.

This module provides:
1. Partitioning/chunking of datasets for fault tolerance
2. Checkpointing support for intermediate data (using Parquet)
3. Event logging system to track partitions and operations
4. Recovery mechanisms for failed partitions
"""

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


@dataclass
class PartitionInfo:
    """Information about a dataset partition."""

    partition_id: int
    start_idx: int
    end_idx: int
    sample_count: int
    file_path: str
    file_size_bytes: int
    checksum: str
    created_timestamp: float
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class OperationCheckpoint:
    """Checkpoint information for an operation."""

    operation_name: str
    operation_idx: int
    partition_id: int
    checkpoint_path: str
    sample_count: int
    file_size_bytes: int
    checksum: str
    timestamp: float
    metadata: Dict[str, Any]


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


class EventLogger:
    """Event logging system for tracking processing operations."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.log_dir / "processing_events.jsonl"
        self.summary_file = self.log_dir / "processing_summary.json"

        # Initialize summary
        self.summary = {
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

    def log_event(self, event: ProcessingEvent):
        """Log a processing event."""
        # Write to JSONL file
        with open(self.events_file, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

        # Update summary
        if event.event_type == "partition_start":
            self.summary["total_partitions"] += 1
        elif event.event_type == "partition_complete":
            self.summary["completed_partitions"] += 1
        elif event.event_type == "partition_failed":
            self.summary["failed_partitions"] += 1
        elif event.event_type == "operation_checkpoint":
            self.summary["checkpoints_created"] += 1
        elif event.event_type == "error":
            self.summary["errors"].append(
                {
                    "timestamp": event.timestamp,
                    "message": event.message,
                    "partition_id": event.partition_id,
                    "operation_name": event.operation_name,
                }
            )

    def finalize_summary(self):
        """Finalize and save the processing summary."""
        self.summary["end_time"] = time.time()
        self.summary["total_processing_time"] = self.summary["end_time"] - self.summary["start_time"]

        with open(self.summary_file, "w") as f:
            json.dump(self.summary, f, indent=2)

    def get_events(self, event_type: Optional[str] = None, partition_id: Optional[int] = None) -> List[ProcessingEvent]:
        """Retrieve events with optional filtering."""
        events = []
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

    def get_partition_status(self, partition_id: int) -> Dict[str, Any]:
        """Get detailed status for a specific partition."""
        events = self.get_events(partition_id=partition_id)

        status = {
            "partition_id": partition_id,
            "events": [asdict(event) for event in events],
            "start_time": None,
            "end_time": None,
            "processing_time": None,
            "status": "unknown",
            "error_count": 0,
        }

        for event in events:
            if event.event_type == "partition_start":
                status["start_time"] = event.timestamp
                status["status"] = "processing"
            elif event.event_type == "partition_complete":
                status["end_time"] = event.timestamp
                status["status"] = "completed"
            elif event.event_type == "partition_failed":
                status["end_time"] = event.timestamp
                status["status"] = "failed"
            elif event.event_type == "error":
                status["error_count"] += 1

        if status["start_time"] and status["end_time"]:
            status["processing_time"] = status["end_time"] - status["start_time"]

        return status


class CheckpointManager:
    """Manages checkpointing of intermediate data."""

    def __init__(self, checkpoint_dir: str, storage_format: str = "parquet"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.storage_format = storage_format
        self.checkpoints: List[OperationCheckpoint] = []

    def create_checkpoint(
        self,
        operation_name: str,
        operation_idx: int,
        partition_id: int,
        data: Union[pd.DataFrame, pa.Table],
        metadata: Dict[str, Any] = None,
    ) -> OperationCheckpoint:
        """Create a checkpoint for intermediate data."""

        # Create checkpoint directory structure
        checkpoint_path = (
            self.checkpoint_dir / f"partition_{partition_id:06d}" / f"op_{operation_idx:03d}_{operation_name}"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save data in specified format
        if self.storage_format == "parquet":
            if isinstance(data, pd.DataFrame):
                data.to_parquet(checkpoint_path.with_suffix(".parquet"), index=False)
            else:  # Arrow table
                pq.write_table(data, checkpoint_path.with_suffix(".parquet"))
            file_path = str(checkpoint_path.with_suffix(".parquet"))
        elif self.storage_format == "arrow":
            if isinstance(data, pd.DataFrame):
                table = pa.Table.from_pandas(data)
            else:
                table = data
            import pyarrow.feather as feather

            feather.write_feather(table, checkpoint_path.with_suffix(".arrow"))
            file_path = str(checkpoint_path.with_suffix(".arrow"))
        else:  # jsonl
            if isinstance(data, pd.DataFrame):
                data.to_json(checkpoint_path.with_suffix(".jsonl"), orient="records", lines=True)
            else:
                df = data.to_pandas()
                df.to_json(checkpoint_path.with_suffix(".jsonl"), orient="records", lines=True)
            file_path = str(checkpoint_path.with_suffix(".jsonl"))

        # Calculate file size and checksum
        file_size = os.path.getsize(file_path)
        checksum = self._calculate_checksum(file_path)

        # Create checkpoint record
        checkpoint = OperationCheckpoint(
            operation_name=operation_name,
            operation_idx=operation_idx,
            partition_id=partition_id,
            checkpoint_path=file_path,
            sample_count=len(data),
            file_size_bytes=file_size,
            checksum=checksum,
            timestamp=time.time(),
            metadata=metadata if metadata is not None else {},
        )

        self.checkpoints.append(checkpoint)
        return checkpoint

    def load_checkpoint(self, checkpoint: OperationCheckpoint) -> Union[pd.DataFrame, pa.Table]:
        """Load data from a checkpoint."""
        if self.storage_format == "parquet":
            return pd.read_parquet(checkpoint.checkpoint_path)
        elif self.storage_format == "arrow":
            import pyarrow.feather as feather

            return feather.read_table(checkpoint.checkpoint_path)
        else:  # jsonl
            return pd.read_json(checkpoint.checkpoint_path, lines=True)

    def get_latest_checkpoint(
        self, partition_id: int, operation_name: Optional[str] = None
    ) -> Optional[OperationCheckpoint]:
        """Get the latest checkpoint for a partition."""
        partition_checkpoints = [c for c in self.checkpoints if c.partition_id == partition_id]

        if operation_name:
            partition_checkpoints = [c for c in partition_checkpoints if c.operation_name == operation_name]

        if not partition_checkpoints:
            return None

        return max(partition_checkpoints, key=lambda c: c.timestamp)

    def cleanup_old_checkpoints(self, keep_latest: int = 1):
        """Clean up old checkpoints, keeping only the latest ones."""
        # Group by partition and operation
        checkpoint_groups = {}
        for checkpoint in self.checkpoints:
            key = (checkpoint.partition_id, checkpoint.operation_name)
            if key not in checkpoint_groups:
                checkpoint_groups[key] = []
            checkpoint_groups[key].append(checkpoint)

        # Keep only the latest checkpoints
        checkpoints_to_keep = []
        for checkpoints in checkpoint_groups.values():
            sorted_checkpoints = sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)
            checkpoints_to_keep.extend(sorted_checkpoints[:keep_latest])

        # Remove old checkpoint files
        checkpoints_to_remove = [c for c in self.checkpoints if c not in checkpoints_to_keep]
        for checkpoint in checkpoints_to_remove:
            try:
                os.remove(checkpoint.checkpoint_path)
            except FileNotFoundError:
                pass

        self.checkpoints = checkpoints_to_keep

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()


class PartitionedExecutorBase(ABC):
    """Base class for partitioned executors with comprehensive checkpointing and event logging."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.event_logger = EventLogger(str(self.work_dir / "logs"))
        self.checkpoint_manager = CheckpointManager(
            str(self.work_dir / "checkpoints"), storage_format=getattr(cfg, "storage_format", "parquet")
        )

        # Partitioning configuration
        self.partition_size = getattr(cfg, "partition_size", 10000)
        self.max_partition_size_mb = getattr(cfg, "max_partition_size_mb", 128)
        self.enable_fault_tolerance = getattr(cfg, "enable_fault_tolerance", True)
        self.max_retries = getattr(cfg, "max_retries", 3)
        self.preserve_intermediate_data = getattr(cfg, "preserve_intermediate_data", False)

        # Initialize partitions
        self.partitions: List[PartitionInfo] = []
        self.dataset_mapping: Dict[str, Any] = {}

    def create_partitions(self, dataset_path: str) -> List[PartitionInfo]:
        """Create partitions from the input dataset."""
        logger.info(f"Creating partitions from {dataset_path}")

        # Calculate total samples and optimal partition size
        total_samples = self._count_samples(dataset_path)
        optimal_partition_size = self._calculate_optimal_partition_size(dataset_path, total_samples)

        # Create partitions
        partitions = []
        partition_id = 0

        for start_idx in range(0, total_samples, optimal_partition_size):
            end_idx = min(start_idx + optimal_partition_size, total_samples)
            sample_count = end_idx - start_idx

            # Create partition file
            partition_path = self._create_partition_file(dataset_path, start_idx, end_idx, partition_id)

            # Calculate file size and checksum
            file_size = os.path.getsize(partition_path)
            checksum = self._calculate_file_checksum(partition_path)

            partition = PartitionInfo(
                partition_id=partition_id,
                start_idx=start_idx,
                end_idx=end_idx,
                sample_count=sample_count,
                file_path=partition_path,
                file_size_bytes=file_size,
                checksum=checksum,
                created_timestamp=time.time(),
            )

            partitions.append(partition)
            partition_id += 1

        self.partitions = partitions

        # Create dataset mapping
        self.dataset_mapping = {
            "original_dataset_path": dataset_path,
            "original_dataset_size": total_samples,
            "partition_count": len(partitions),
            "partition_size": optimal_partition_size,
            "created_timestamp": time.time(),
            "partitions": [asdict(p) for p in partitions],
        }

        # Save dataset mapping
        mapping_path = self.work_dir / "metadata" / "dataset_mapping.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump(self.dataset_mapping, f, indent=2)

        logger.info(f"Created {len(partitions)} partitions with {optimal_partition_size} samples each")
        return partitions

    def process_partition(self, partition: PartitionInfo, operations: List[Dict[str, Any]]) -> bool:
        """Process a single partition through the pipeline."""
        partition_id = partition.partition_id

        # Log partition start
        self.event_logger.log_event(
            ProcessingEvent(
                event_id=f"partition_{partition_id}_start_{int(time.time())}",
                event_type="partition_start",
                timestamp=time.time(),
                partition_id=partition_id,
                message=f"Starting processing of partition {partition_id}",
            )
        )

        partition.processing_start_time = time.time()
        partition.processing_status = "processing"

        try:
            # Load partition data
            data = self._load_partition_data(partition)

            # Process through operations
            for op_idx, operation in enumerate(operations):
                op_name = list(operation.keys())[0]
                op_config = operation[op_name]

                # Log operation start
                self.event_logger.log_event(
                    ProcessingEvent(
                        event_id=f"op_{partition_id}_{op_idx}_start_{int(time.time())}",
                        event_type="operation_start",
                        timestamp=time.time(),
                        partition_id=partition_id,
                        operation_name=op_name,
                        operation_idx=op_idx,
                        message=f"Starting operation {op_name} on partition {partition_id}",
                    )
                )

                # Apply operation
                data = self._apply_operation(data, op_name, op_config)

                # Create checkpoint if enabled
                if self.preserve_intermediate_data:
                    _ = self.checkpoint_manager.create_checkpoint(
                        operation_name=op_name,
                        operation_idx=op_idx,
                        partition_id=partition_id,
                        data=data,
                        metadata={"operation_config": op_config},
                    )

                    self.event_logger.log_event(
                        ProcessingEvent(
                            event_id=f"checkpoint_{partition_id}_{op_idx}_{int(time.time())}",
                            event_type="operation_checkpoint",
                            timestamp=time.time(),
                            partition_id=partition_id,
                            operation_name=op_name,
                            operation_idx=op_idx,
                            message=f"Created checkpoint for {op_name} on partition {partition_id}",
                        )
                    )

                # Log operation completion
                self.event_logger.log_event(
                    ProcessingEvent(
                        event_id=f"op_{partition_id}_{op_idx}_complete_{int(time.time())}",
                        event_type="operation_complete",
                        timestamp=time.time(),
                        partition_id=partition_id,
                        operation_name=op_name,
                        operation_idx=op_idx,
                        message=f"Completed operation {op_name} on partition {partition_id}",
                    )
                )

            # Save final result
            self._save_partition_result(partition, data)

            # Update partition status
            partition.processing_end_time = time.time()
            partition.processing_status = "completed"

            # Log partition completion
            self.event_logger.log_event(
                ProcessingEvent(
                    event_id=f"partition_{partition_id}_complete_{int(time.time())}",
                    event_type="partition_complete",
                    timestamp=time.time(),
                    partition_id=partition_id,
                    message=f"Completed processing of partition {partition_id}",
                )
            )

            return True

        except Exception as e:
            # Handle failure
            partition.processing_end_time = time.time()
            partition.processing_status = "failed"
            partition.error_message = str(e)
            partition.retry_count += 1

            # Log error
            self.event_logger.log_event(
                ProcessingEvent(
                    event_id=f"partition_{partition_id}_error_{int(time.time())}",
                    event_type="partition_failed",
                    timestamp=time.time(),
                    partition_id=partition_id,
                    message=f"Failed to process partition {partition_id}: {str(e)}",
                    error_details=str(e),
                )
            )

            logger.error(f"Failed to process partition {partition_id}: {e}")
            return False

    def recover_failed_partitions(self, operations: List[Dict[str, Any]]) -> List[PartitionInfo]:
        """Recover failed partitions using checkpoints."""
        failed_partitions = [p for p in self.partitions if p.processing_status == "failed"]
        recovered_partitions = []

        for partition in failed_partitions:
            if partition.retry_count >= self.max_retries:
                logger.warning(f"Partition {partition.partition_id} exceeded max retries")
                continue

            # Find latest checkpoint
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(partition.partition_id)

            if latest_checkpoint:
                logger.info(
                    f"Recovering partition {partition.partition_id} from checkpoint {latest_checkpoint.operation_name}"
                )

                # Load from checkpoint and continue processing
                _ = self.checkpoint_manager.load_checkpoint(latest_checkpoint)

                # Continue processing from the next operation
                remaining_operations = operations[latest_checkpoint.operation_idx + 1 :]

                if self.process_partition(partition, remaining_operations):
                    recovered_partitions.append(partition)
            else:
                # No checkpoint available, restart from beginning
                logger.info(f"Restarting partition {partition.partition_id} from beginning")
                if self.process_partition(partition, operations):
                    recovered_partitions.append(partition)

        return recovered_partitions

    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final processing report."""
        completed_partitions = [p for p in self.partitions if p.processing_status == "completed"]
        failed_partitions = [p for p in self.partitions if p.processing_status == "failed"]

        total_processed_samples = sum(p.sample_count for p in completed_partitions)

        report = {
            "original_dataset": {
                "path": self.dataset_mapping["original_dataset_path"],
                "total_samples": self.dataset_mapping["original_dataset_size"],
                "partition_count": self.dataset_mapping["partition_count"],
            },
            "processing_summary": {
                "total_partitions": len(self.partitions),
                "successful_partitions": len(completed_partitions),
                "failed_partitions": len(failed_partitions),
                "total_processed_samples": total_processed_samples,
                "success_rate": len(completed_partitions) / len(self.partitions) if self.partitions else 0,
                "total_processing_time": time.time() - self.event_logger.summary["start_time"],
                "checkpoints_created": len(self.checkpoint_manager.checkpoints),
            },
            "partition_details": [asdict(p) for p in self.partitions],
            "checkpoints": [asdict(c) for c in self.checkpoint_manager.checkpoints],
        }

        # Save report
        report_path = self.work_dir / "metadata" / "final_mapping_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    @abstractmethod
    def _count_samples(self, dataset_path: str) -> int:
        """Count total samples in dataset."""
        pass

    @abstractmethod
    def _calculate_optimal_partition_size(self, dataset_path: str, total_samples: int) -> int:
        """Calculate optimal partition size based on dataset characteristics."""
        pass

    @abstractmethod
    def _create_partition_file(self, dataset_path: str, start_idx: int, end_idx: int, partition_id: int) -> str:
        """Create a partition file from the dataset."""
        pass

    @abstractmethod
    def _load_partition_data(self, partition: PartitionInfo) -> Union[pd.DataFrame, pa.Table]:
        """Load data from a partition file."""
        pass

    @abstractmethod
    def _apply_operation(
        self, data: Union[pd.DataFrame, pa.Table], op_name: str, op_config: Dict[str, Any]
    ) -> Union[pd.DataFrame, pa.Table]:
        """Apply an operation to the data."""
        pass

    @abstractmethod
    def _save_partition_result(self, partition: PartitionInfo, data: Union[pd.DataFrame, pa.Table]):
        """Save the final result for a partition."""
        pass

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
