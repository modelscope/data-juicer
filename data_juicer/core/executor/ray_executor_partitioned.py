"""
Simplified Partitioned Ray Executor for Large Dataset Processing

This module implements a streamlined partitioned execution strategy for Ray mode that:
2. Splits the dataset into manageable partitions using Ray's .split() method
3. Processes each partition independently with Ray tasks
4. Merges results back into a single dataset for export
5. Supports convergence points for global operations (like deduplicators)
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.adapter import Adapter
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")


# Note: Using Ray Data's built-in map_batches for parallel processing instead of custom remote functions


class CheckpointStrategy(Enum):
    """Checkpoint strategies for controlling when to create checkpoints."""

    EVERY_OP = "every_op"  # Checkpoint after every operation
    EVERY_N_OPS = "every_n_ops"  # Checkpoint after every N operations
    MANUAL = "manual"  # Checkpoint only after specified operations
    DISABLED = "disabled"  # Disable checkpointing entirely


# Simplified classes for basic functionality
@dataclass
class PartitionResult:
    """Simple result container for partition processing."""

    partition_id: int
    dataset: Optional[Any] = None
    success: bool = False
    error: Optional[str] = None


class PartitionedRayExecutor(ExecutorBase, EventLoggingMixin, DAGExecutionMixin):
    """
    Simplified Ray executor with dataset partitioning using .split().

    Features:
    - Single DatasetBuilder loads the full dataset
    - Uses Ray's .split() method for partitioning
    - Processes partitions in parallel with Ray tasks
    - Supports convergence points for global operations
    - Merges results back into a single dataset
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """Initialize the partitioned Ray executor."""
        super().__init__(cfg)

        self.executor_type = "ray_partitioned"
        self.work_dir = self.cfg.work_dir
        self.adapter = Adapter(self.cfg)

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)

        # Override strategy methods for partitioned execution
        self._override_strategy_methods()

        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        # Partition configuration
        self.num_partitions = getattr(self.cfg, "num_partitions", 4)  # Default to 4 partitions
        logger.info(f"Using dataset splitting with {self.num_partitions} partitions")

        # Checkpoint configuration
        checkpoint_cfg = getattr(self.cfg, "checkpoint", None)
        if checkpoint_cfg:
            # Handle both dict and object configurations
            if isinstance(checkpoint_cfg, dict):
                self.checkpoint_enabled = checkpoint_cfg.get("enabled", True)
                strategy_str = checkpoint_cfg.get("strategy", "every_op")
                self.checkpoint_n_ops = checkpoint_cfg.get("n_ops", 1)
                self.checkpoint_op_names = set(checkpoint_cfg.get("op_names", []))
            else:
                self.checkpoint_enabled = getattr(checkpoint_cfg, "enabled", True)
                strategy_str = getattr(checkpoint_cfg, "strategy", "every_op")
                self.checkpoint_n_ops = getattr(checkpoint_cfg, "n_ops", 1)
                self.checkpoint_op_names = set(getattr(checkpoint_cfg, "op_names", []))

            # Parse checkpoint strategy with validation
            try:
                self.checkpoint_strategy = CheckpointStrategy(strategy_str)
            except ValueError:
                logger.warning(f"Unknown checkpoint strategy: {strategy_str}, defaulting to EVERY_OP")
                self.checkpoint_strategy = CheckpointStrategy.EVERY_OP
        else:
            self.checkpoint_enabled = False
            self.checkpoint_strategy = CheckpointStrategy.DISABLED
            self.checkpoint_n_ops = 1
            self.checkpoint_op_names = set()

        # If strategy is DISABLED, disable checkpointing regardless of enabled flag
        if self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            self.checkpoint_enabled = False

        # Checkpoint directory
        self.checkpoint_dir = getattr(self.cfg, "checkpoint_dir", os.path.join(self.work_dir, "checkpoints"))
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info(f"Checkpointing: {'enabled' if self.checkpoint_enabled else 'disabled'}")
        if self.checkpoint_enabled:
            logger.info(f"Checkpoint strategy: {self.checkpoint_strategy.value}")
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        # Initialize RayExporter for final output
        logger.info("Preparing exporter...")
        self.exporter = RayExporter(
            self.cfg.export_path,
            keep_stats_in_res_ds=getattr(self.cfg, "keep_stats_in_res_ds", True),
            keep_hashes_in_res_ds=getattr(self.cfg, "keep_hashes_in_res_ds", False),
        )

    def _should_checkpoint(self, op_idx: int, op_name: str) -> bool:
        """Determine if checkpoint should be created based on configuration strategy."""
        if not self.checkpoint_enabled:
            return False

        if self.checkpoint_strategy == CheckpointStrategy.EVERY_OP:
            return True
        elif self.checkpoint_strategy == CheckpointStrategy.EVERY_N_OPS:
            return (op_idx + 1) % self.checkpoint_n_ops == 0
        elif self.checkpoint_strategy == CheckpointStrategy.MANUAL:
            return op_name in self.checkpoint_op_names
        elif self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            return False
        else:
            logger.warning(f"Unknown checkpoint strategy: {self.checkpoint_strategy}, defaulting to every_op")
            return True

    def _save_checkpoint(self, dataset: RayDataset, op_idx: int, op_name: str, partition_id: int = 0) -> str:
        """Save dataset checkpoint to parquet format."""
        checkpoint_filename = f"checkpoint_op_{op_idx:03d}_{op_name}_partition_{partition_id:03d}.parquet"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Save as parquet
        dataset.data.write_parquet(checkpoint_path)

        # Log checkpoint save event
        self._log_event(
            event_type=EventType.CHECKPOINT_SAVE,
            message=f"Saved checkpoint after operation {op_idx}: {op_name}",
            partition_id=partition_id,
            operation_name=op_name,
            operation_idx=op_idx,
            metadata={"checkpoint_path": checkpoint_path},
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def _load_checkpoint(self, op_idx: int, op_name: str, partition_id: int = 0) -> Optional[RayDataset]:
        """Load dataset checkpoint from parquet format."""
        checkpoint_filename = f"checkpoint_op_{op_idx:03d}_{op_name}_partition_{partition_id:03d}.parquet"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            return None

        try:
            # Load from parquet
            ray_dataset = ray.data.read_parquet(checkpoint_path)

            # Log checkpoint load event
            self._log_event(
                event_type=EventType.CHECKPOINT_LOAD,
                message=f"Loaded checkpoint from operation {op_idx}: {op_name}",
                partition_id=partition_id,
                operation_name=op_name,
                operation_idx=op_idx,
                metadata={"checkpoint_path": checkpoint_path},
            )

            return RayDataset(ray_dataset, cfg=self.cfg)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None

    def _find_latest_checkpoint(self, partition_id: int = 0) -> Optional[Tuple[int, str, str]]:
        """Find the latest checkpoint for a partition. Returns (op_idx, op_name, checkpoint_path)."""
        checkpoint_files = []

        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(f"checkpoint_op_") and filename.endswith(f"_partition_{partition_id:03d}.parquet"):
                try:
                    # Parse filename: checkpoint_op_XXX_OpName_partition_YYY.parquet
                    parts = filename.replace(".parquet", "").split("_")
                    if len(parts) >= 5:
                        op_idx = int(parts[2])
                        op_name = parts[3]
                        checkpoint_files.append((op_idx, op_name, os.path.join(self.checkpoint_dir, filename)))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_files:
            return None

        # Return the latest checkpoint (highest op_idx)
        latest = max(checkpoint_files, key=lambda x: x[0])
        return latest

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Run the simplified partitioned dataset processing pipeline.

        Args:
            load_data_np: Number of workers for loading dataset
            skip_return: Whether to skip returning the dataset

        Returns:
            Processed dataset
        """
        job_start_time = time.time()
        logger.info("🚀 Starting simplified partitioned processing...")

        # Log job start event
        self._log_event(
            event_type=EventType.JOB_START,
            message="Starting partitioned dataset processing",
            metadata={"num_partitions": self.num_partitions, "checkpoint_enabled": self.checkpoint_enabled},
        )

        # Load the full dataset using a single DatasetBuilder
        logger.info("Loading dataset with single DatasetBuilder...")

        dataset = self.datasetbuilder.load_dataset(num_proc=load_data_np)
        columns = dataset.schema().columns

        # Prepare operations
        logger.info("Preparing operations...")
        ops = self._prepare_operators()

        # Detect convergence points for global operations
        convergence_points = self._detect_convergence_points_partitioned(ops)

        if convergence_points:
            logger.info(f"Found convergence points at operations: {convergence_points}")
            final_dataset = self._process_with_convergence(dataset, ops, convergence_points)
        else:
            logger.info("No convergence points found, processing with simple partitioning")
            final_dataset = self._process_with_simple_partitioning(dataset, ops)

        # Export final dataset
        logger.info("Exporting final dataset...")
        self.exporter.export(final_dataset.data, columns=columns)

        job_duration = time.time() - job_start_time
        logger.info(f"✅ Job completed successfully in {job_duration:.2f}s")
        logger.info(f"📁 Output saved to: {self.cfg.export_path}")

        # Log job completion event
        self._log_event(
            event_type=EventType.JOB_COMPLETE,
            message="Partitioned dataset processing completed successfully",
            metadata={"duration_seconds": job_duration, "export_path": self.cfg.export_path},
        )

        if skip_return:
            return None

        return final_dataset

    def _process_with_simple_partitioning(self, dataset: RayDataset, ops: List):
        """
        Process dataset with real partitioning using Ray Data's split and union.
        """
        logger.info("Processing with real partitioning using Ray Data's split and union...")

        # Split the dataset into partitions
        logger.info(f"Splitting dataset into {self.num_partitions} partitions...")
        partitions = dataset.data.split(self.num_partitions)
        logger.info(f"Created {len(partitions)} partitions")

        # Process each partition separately with checkpointing
        logger.info("Processing partitions with checkpointing support...")
        processed_partitions = []

        for i, partition in enumerate(partitions):
            logger.info(f"Processing partition {i+1}/{len(partitions)}")

            # Log partition start event
            self._log_event(
                event_type=EventType.PARTITION_START,
                message=f"Starting processing of partition {i+1}/{len(partitions)}",
                partition_id=i,
            )

            # Create a RayDataset wrapper for this partition
            partition_dataset = RayDataset(partition, cfg=self.cfg)

            # Apply operations with checkpointing support
            processed_partition = self._process_with_checkpointing(partition_dataset, i, ops)

            # Store the processed partition's data
            processed_partitions.append(processed_partition.data)

            # Log partition completion event
            self._log_event(
                event_type=EventType.PARTITION_COMPLETE,
                message=f"Completed processing of partition {i+1}/{len(partitions)}",
                partition_id=i,
            )

        # Merge all processed partitions back into a single dataset
        logger.info("Merging processed partitions...")
        if len(processed_partitions) == 1:
            merged_dataset = processed_partitions[0]
        else:
            # Union all partitions
            merged_dataset = processed_partitions[0]
            for partition in processed_partitions[1:]:
                merged_dataset = merged_dataset.union(partition)

        # Return as RayDataset wrapper
        return RayDataset(merged_dataset, cfg=self.cfg)

    def _process_with_convergence(self, dataset: RayDataset, ops: List, convergence_points: List[int]):
        """
        Process dataset with convergence support for global operations.
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
            processed_dataset = self._process_with_simple_partitioning(dataset, pre_convergence_ops)
        else:
            logger.info("No pre-convergence operations, using original dataset...")
            processed_dataset = dataset

        # Merge partitions for global operations
        logger.info("Merging partitions for global operations...")
        merged_dataset = processed_dataset.data

        # Process merged dataset with post-convergence operations
        if post_convergence_ops:
            logger.info("Processing merged dataset with global operations...")
            merged_ray_dataset = RayDataset(merged_dataset, cfg=self.cfg)
            final_dataset = merged_ray_dataset.process(post_convergence_ops)
            logger.info("Global operations completed. Final dataset ready for export")
            return final_dataset
        else:
            # No post-convergence operations, just return the merged result
            return RayDataset(merged_dataset, cfg=self.cfg)

    def _process_with_checkpointing(self, dataset: RayDataset, partition_id: int, ops: List) -> RayDataset:
        """
        Process dataset with checkpointing support.
        Groups operations and checkpoints between groups based on strategy.
        """
        logger.info(f"Processing partition {partition_id} with checkpointing support...")

        if not self.checkpoint_enabled:
            logger.info(f"Checkpointing disabled, processing all operations at once for partition {partition_id}")
            return dataset.process(ops)

        # check the latest checkpoint for the partition
        latest_checkpoint = self._find_latest_checkpoint(partition_id)

        # Group operations based on checkpoint strategy
        op_groups = self._group_operations_for_checkpointing(ops)
        logger.info(f"Grouped {len(ops)} operations into {len(op_groups)} groups for checkpointing")
        logger.info(f"Detailed op gruops: {op_groups}")

        current_dataset = dataset

        for group_idx, (start_idx, end_idx, group_ops) in enumerate(op_groups):
            logger.info(
                f"Processing partition {partition_id}, group {group_idx + 1}/{len(op_groups)}: operations {start_idx}-{end_idx-1}"
            )

            if latest_checkpoint and latest_checkpoint[0] >= end_idx:
                logger.info(
                    f"Partition {partition_id}: All operations in group {group_idx + 1} already processed (checkpoint at op {latest_checkpoint[0]}, group ends at {end_idx-1}), skipping"
                )
                continue

            if latest_checkpoint and latest_checkpoint[0] >= start_idx:
                logger.info(f"Partition {partition_id}: Resuming from checkpoint at operation {latest_checkpoint[0]}")
                current_dataset = self._load_checkpoint(latest_checkpoint[0], latest_checkpoint[1], partition_id)
                if current_dataset is None:
                    logger.warning(f"Partition {partition_id}: Failed to load checkpoint, starting from beginning")
                    current_dataset = dataset
                group_ops = ops[latest_checkpoint[0] + 1 : end_idx]
                if not group_ops:
                    logger.info(f"Partition {partition_id}: All operations in this group already processed, skipping")
                    continue

            # Process the group of operations
            if group_ops:
                logger.info(
                    f"Partition {partition_id}: Processing {len(group_ops)} operations in group {group_idx + 1}"
                )

                # Log operation start events
                for op_idx, op in enumerate(group_ops):
                    self._log_event(
                        event_type=EventType.OP_START,
                        message=f"Starting operation: {op._name}",
                        operation_name=op._name,
                        operation_idx=start_idx + op_idx,
                        partition_id=partition_id,
                    )

                current_dataset = current_dataset.process(group_ops)

                # Log operation completion events
                for op_idx, op in enumerate(group_ops):
                    self._log_event(
                        event_type=EventType.OP_COMPLETE,
                        message=f"Completed operation: {op._name}",
                        operation_name=op._name,
                        operation_idx=start_idx + op_idx,
                        partition_id=partition_id,
                    )

            # Checkpoint after the last operation in the group
            if group_ops:
                last_op_idx = end_idx - 1
                last_op_name = ops[last_op_idx]._name
                if self._should_checkpoint(last_op_idx, last_op_name):
                    logger.info(
                        f"Partition {partition_id}: Creating checkpoint after operation {last_op_idx}: {last_op_name}"
                    )
                    self._save_checkpoint(current_dataset, last_op_idx, last_op_name, partition_id)

        return current_dataset

    def _group_operations_for_checkpointing(self, ops: List) -> List[Tuple[int, int, List]]:
        """
        Group operations based on checkpoint strategy.
        Returns list of (start_idx, end_idx, group_ops) tuples.
        """
        groups = []
        current_start = 0

        for i, op in enumerate(ops):
            if self._should_checkpoint(i, op._name):
                # This operation should trigger a checkpoint
                groups.append((current_start, i + 1, ops[current_start : i + 1]))
                current_start = i + 1

        # Add remaining operations as the last group
        if current_start < len(ops):
            groups.append((current_start, len(ops), ops[current_start:]))

        return groups

    def _prepare_operators(self):
        """Prepare process operators."""
        ops = load_ops(self.cfg.process)

        # Check for op_fusion configuration with safe attribute access
        if hasattr(self.cfg, "op_fusion") and self.cfg.op_fusion:
            probe_res = None
            fusion_strategy = getattr(self.cfg, "fusion_strategy", "basic")
            if fusion_strategy == "probe":
                logger.info("Probe the OP speed for OP reordering...")
                probe_res, _ = self.adapter.probe_small_batch(self.dataset, ops)

            logger.info(f"Start OP fusion and reordering with strategy [{fusion_strategy}]...")
            ops = fuse_operators(ops, probe_res)

        return ops

    def _override_strategy_methods(self):
        """Override strategy methods for partitioned execution."""
        # Override partition count determination
        self._determine_partition_count = self._determine_partition_count_partitioned
        self._analyze_dataset_size = self._analyze_dataset_size_partitioned
        self._detect_convergence_points = self._detect_convergence_points_partitioned
        self._get_dag_node_for_operation = self._get_dag_node_for_operation_partitioned

    def _determine_partition_count_partitioned(self, cfg) -> int:
        """Determine partition count for partitioned execution."""
        return self.num_partitions

    def _analyze_dataset_size_partitioned(self, dataset_path: str) -> int:
        """Analyze dataset size for partition count determination."""
        try:
            file_size = os.path.getsize(dataset_path)
            # More accurate estimate for partitioned execution
            estimated_lines = file_size // 512  # Assume 512 bytes per line
            return estimated_lines
        except Exception as e:
            logger.error(f"Error analyzing dataset size: {e}")
            # Fallback to default
            return 100000

    def _detect_convergence_points_partitioned(self, operations: List) -> List[int]:
        """Detect convergence points for partitioned execution."""
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
