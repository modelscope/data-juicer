#!/usr/bin/env python3
"""
Event Logging Mixin for Data-Juicer Executors

This module provides comprehensive event logging capabilities that can be used
by any executor (default, ray, partitioned, etc.) to track operations,
performance, and errors in real-time.

Features:
1. Real-time event logging with configurable levels
2. Event filtering and querying
3. Performance metrics tracking
4. Error tracking with stack traces
5. Status reporting and monitoring
6. Log rotation and cleanup
"""

import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

from loguru import logger


class EventType(Enum):
    """Types of events that can be logged."""

    JOB_START = "job_start"
    JOB_COMPLETE = "job_complete"
    JOB_FAILED = "job_failed"
    PARTITION_START = "partition_start"
    PARTITION_COMPLETE = "partition_complete"
    PARTITION_FAILED = "partition_failed"
    OP_START = "op_start"
    OP_COMPLETE = "op_complete"
    OP_FAILED = "op_failed"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    PROCESSING_START = "processing_start"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_METRIC = "performance_metric"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class Event:
    """Event data structure."""

    event_type: EventType
    timestamp: float
    message: str
    job_id: Optional[str] = None
    partition_id: Optional[int] = None
    operation_name: Optional[str] = None
    operation_idx: Optional[int] = None
    status: Optional[str] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    retry_count: Optional[int] = None
    checkpoint_path: Optional[str] = None
    op_args: Optional[Dict[str, Any]] = None
    input_rows: Optional[int] = None
    output_rows: Optional[int] = None
    output_path: Optional[str] = None
    partition_meta: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    resource_usage: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class EventLogger:
    """Event logging system with real-time capabilities and JSONL event log for resumability."""

    def __init__(self, log_dir: str, max_log_size_mb: int = 100, backup_count: int = 5, job_id: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_log_size_mb = max_log_size_mb
        self.backup_count = backup_count
        # Use provided job_id or generate a simple timestamp-based one
        self.job_id = job_id or f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}-{uuid4().hex[:6]}"
        self.events: deque = deque(maxlen=10000)
        self.event_lock = threading.Lock()
        self.performance_metrics = defaultdict(list)
        self.resource_usage = defaultdict(list)
        self._setup_file_logging()
        self._start_cleanup_thread()
        # Use simpler filename since we're in job-specific directory
        self.jsonl_file = self.log_dir / "events.jsonl"

    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = self.log_dir / "events.log"

        # Configure loguru for file logging
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation=f"{self.max_log_size_mb} MB",
            retention=self.backup_count,  # Use number directly, not string
            compression="gz",
        )

    def _start_cleanup_thread(self):
        """Start background thread for log cleanup."""

        def cleanup_worker():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._cleanup_old_logs()
                except Exception as e:
                    logger.warning(f"Error in cleanup thread: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_old_logs(self):
        """Clean up old log files."""
        try:
            log_files = list(self.log_dir.glob("events.log.*"))
            if len(log_files) > self.backup_count:
                # Sort by modification time and remove oldest
                log_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in log_files[: -self.backup_count]:
                    old_file.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up old logs: {e}")

    def log_event(self, event: Event):
        """Log an event (to memory, loguru, and JSONL for resumability)."""
        with self.event_lock:
            event.job_id = self.job_id
            self.events.append(event)
            # Log to file (loguru)
            log_message = self._format_event_for_logging(event)
            logger.info(log_message)
            # Write to JSONL for resumability
            with open(self.jsonl_file, "a") as f:
                f.write(
                    json.dumps(
                        {k: (v.value if isinstance(v, Enum) else v) for k, v in event.__dict__.items() if v is not None}
                    )
                    + "\n"
                )
            # Track performance metrics
            if event.performance_metrics:
                self.performance_metrics[event.operation_name or "unknown"].append(event.performance_metrics)
            if event.resource_usage:
                self.resource_usage[event.operation_name or "unknown"].append(event.resource_usage)

    def _format_event_for_logging(self, event: Event) -> str:
        """Format event for logging."""
        parts = [f"EVENT[{event.event_type.value}]", f"TIME[{datetime.fromtimestamp(event.timestamp).isoformat()}]"]

        if event.partition_id is not None:
            parts.append(f"PARTITION[{event.partition_id}]")

        if event.operation_name:
            parts.append(f"OP[{event.operation_name}]")

        if event.duration is not None:
            parts.append(f"DURATION[{event.duration:.3f}s]")

        parts.append(f"MSG[{event.message}]")

        if event.error_message:
            parts.append(f"ERROR[{event.error_message}]")

        if event.metadata:
            parts.append(f"META[{json.dumps(event.metadata)}]")

        return " | ".join(parts)

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        partition_id: Optional[int] = None,
        operation_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Get events with optional filtering."""
        with self.event_lock:
            filtered_events = []

            for event in self.events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                if partition_id is not None and event.partition_id != partition_id:
                    continue
                if operation_name and event.operation_name != operation_name:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue

                filtered_events.append(event)

            # Apply limit
            if limit:
                filtered_events = filtered_events[-limit:]

            return filtered_events

    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        with self.event_lock:
            if operation_name:
                metrics = self.performance_metrics.get(operation_name, [])
            else:
                # Combine all metrics
                metrics = []
                for op_metrics in self.performance_metrics.values():
                    metrics.extend(op_metrics)

            if not metrics:
                return {}

            # Calculate statistics
            durations = [m.get("duration", 0) for m in metrics]
            throughputs = [m.get("throughput", 0) for m in metrics]

            return {
                "total_operations": len(metrics),
                "avg_duration": sum(durations) / len(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
                "total_throughput": sum(throughputs) if throughputs else 0,
            }

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        with self.event_lock:
            all_usage = []
            for op_usage in self.resource_usage.values():
                all_usage.extend(op_usage)

            if not all_usage:
                return {}

            # Calculate statistics
            cpu_usage = [u.get("cpu_percent", 0) for u in all_usage]
            memory_usage = [u.get("memory_mb", 0) for u in all_usage]

            return {
                "avg_cpu_percent": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "max_cpu_percent": max(cpu_usage) if cpu_usage else 0,
                "avg_memory_mb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "max_memory_mb": max(memory_usage) if memory_usage else 0,
            }

    def generate_status_report(self) -> str:
        """Generate a comprehensive status report."""
        with self.event_lock:
            total_events = len(self.events)
            if total_events == 0:
                return "No events logged yet."

            # Count event types
            event_counts = defaultdict(int)
            error_count = 0
            warning_count = 0

            for event in self.events:
                event_counts[event.event_type.value] += 1
                if event.event_type == EventType.OPERATION_ERROR:
                    error_count += 1
                elif event.event_type == EventType.WARNING:
                    warning_count += 1

            # Get performance summary
            perf_summary = self.get_performance_summary()
            resource_summary = self.get_resource_summary()

            # Generate report
            report_lines = [
                "=== EVENT LOGGING STATUS REPORT ===",
                f"Total Events: {total_events}",
                f"Errors: {error_count}",
                f"Warnings: {warning_count}",
                "",
                "Event Type Distribution:",
            ]

            for event_type, count in sorted(event_counts.items()):
                percentage = (count / total_events) * 100
                report_lines.append(f"  {event_type}: {count} ({percentage:.1f}%)")

            if perf_summary:
                report_lines.extend(
                    [
                        "",
                        "Performance Summary:",
                        f"  Total Operations: {perf_summary.get('total_operations', 0)}",
                        f"  Average Duration: {perf_summary.get('avg_duration', 0):.3f}s",
                        f"  Average Throughput: {perf_summary.get('avg_throughput', 0):.1f} samples/s",
                    ]
                )

            if resource_summary:
                report_lines.extend(
                    [
                        "",
                        "Resource Usage Summary:",
                        f"  Average CPU: {resource_summary.get('avg_cpu_percent', 0):.1f}%",
                        f"  Max CPU: {resource_summary.get('max_cpu_percent', 0):.1f}%",
                        f"  Average Memory: {resource_summary.get('avg_memory_mb', 0):.1f} MB",
                        f"  Max Memory: {resource_summary.get('max_memory_mb', 0):.1f} MB",
                    ]
                )

            return "\n".join(report_lines)

    def monitor_events(self, event_type: Optional[EventType] = None) -> Generator[Event, None, None]:
        """Monitor events in real-time."""
        last_event_count = len(self.events)

        while True:
            with self.event_lock:
                current_events = list(self.events)

            # Yield new events
            for event in current_events[last_event_count:]:
                if event_type is None or event.event_type == event_type:
                    yield event

            last_event_count = len(current_events)
            time.sleep(0.1)  # Check every 100ms

    @classmethod
    def list_available_jobs(cls, work_dir: str) -> List[Dict[str, Any]]:
        """List available jobs for resumption from a work directory."""
        available_jobs = []

        if not os.path.exists(work_dir):
            return available_jobs

        # Look for job directories (each job has its own directory)
        for item in os.listdir(work_dir):
            job_dir = os.path.join(work_dir, item)
            if os.path.isdir(job_dir):
                summary_file = os.path.join(job_dir, "job_summary.json")
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, "r") as f:
                            job_summary = json.load(f)
                        job_summary["work_dir"] = work_dir
                        job_summary["job_dir"] = job_dir
                        available_jobs.append(job_summary)
                    except Exception as e:
                        logger.warning(f"Failed to load job summary from {summary_file}: {e}")

        return available_jobs


class EventLoggingMixin:
    """Mixin to add event logging capabilities to any executor."""

    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        # Initialize event logging if not already done
        if not hasattr(self, "event_logger"):
            self._setup_event_logging()

    def _setup_event_logging(self):
        """Setup event logging for the executor."""
        # Get event logging configuration
        event_config = getattr(self.cfg, "event_logging", {})
        enabled = event_config.get("enabled", True)

        if not enabled:
            self.event_logger = None
            return

        # Use job_id from config if provided, otherwise auto-generate
        job_id = getattr(self.cfg, "job_id", None)

        # Create job-specific directory structure
        if job_id:
            # Use provided job_id
            job_dir = os.path.join(self.work_dir, job_id)
        else:
            # Auto-generate job_id with timestamp and config name
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            config_name = self._get_config_name()
            unique_suffix = uuid4().hex[:6]
            job_id = f"{timestamp}_{config_name}_{unique_suffix}"
            job_dir = os.path.join(self.work_dir, job_id)

        # Create job directory and subdirectories
        os.makedirs(job_dir, exist_ok=True)

        # Determine event log directory - use configured path if available
        event_log_dir = getattr(self.cfg, "event_log_dir", None)
        if event_log_dir:
            # Use configured event log directory with job subdirectory
            event_logs_dir = os.path.join(event_log_dir, job_id, "event_logs")
        else:
            # Use job-specific directory
            event_logs_dir = os.path.join(job_dir, "event_logs")

        os.makedirs(event_logs_dir, exist_ok=True)

        # Setup event logger in the determined directory
        max_log_size = event_config.get("max_log_size_mb", 100)
        backup_count = event_config.get("backup_count", 5)

        self.event_logger = EventLogger(event_logs_dir, max_log_size, backup_count, job_id=job_id)

        # If job_id is provided, validate resumption
        if getattr(self.cfg, "job_id", None):
            if not self._validate_job_resumption(job_id):
                logger.warning("Job resumption validation failed, continuing with new job")

        # Create and display job summary
        self._create_job_summary(job_id, job_dir)

        # Log initialization
        self._log_event(EventType.INFO, f"Event logging initialized for {self.executor_type} executor")

    def _create_job_summary(self, job_id: str, job_dir: str):
        """Create and display job summary for easy resumption."""
        # Get separate storage paths if configured
        event_log_dir = getattr(self.cfg, "event_log_dir", None)
        checkpoint_dir = getattr(self.cfg, "checkpoint_dir", None)

        # Use configured paths or fall back to job-specific directories
        if event_log_dir:
            # Use configured event log directory with job subdirectory
            job_event_log_dir = os.path.join(event_log_dir, job_id, "event_logs")
        else:
            # Use job-specific directory
            job_event_log_dir = os.path.join(job_dir, "event_logs")

        if checkpoint_dir:
            # Use configured checkpoint directory with job subdirectory
            job_checkpoint_dir = os.path.join(checkpoint_dir, job_id)
        else:
            # Use job-specific directory
            job_checkpoint_dir = os.path.join(job_dir, "checkpoints")

        job_metadata_dir = os.path.join(job_dir, "metadata")

        job_summary = {
            "job_id": job_id,
            "start_time": time.time(),
            "work_dir": self.work_dir,
            "job_dir": job_dir,
            "config_file": getattr(self.cfg, "config", None),
            "executor_type": getattr(self, "executor_type", "unknown"),
            "status": "running",
            "resumption_command": f"dj-process --config {getattr(self.cfg, 'config', 'config.yaml')} --job_id {job_id}",
            "event_log_file": os.path.join(job_event_log_dir, "events.jsonl"),
            "event_log_dir": job_event_log_dir,
            "checkpoint_dir": job_checkpoint_dir,
            "metadata_dir": job_metadata_dir,
            "storage_config": {
                "event_log_dir": event_log_dir,
                "checkpoint_dir": checkpoint_dir,
            },
        }

        # Create directories
        os.makedirs(job_event_log_dir, exist_ok=True)
        os.makedirs(job_checkpoint_dir, exist_ok=True)
        os.makedirs(job_metadata_dir, exist_ok=True)

        # Save job summary in job-specific directory
        summary_file = os.path.join(job_dir, "job_summary.json")
        with open(summary_file, "w") as f:
            json.dump(job_summary, f, indent=2, default=str)

        # Display job info to user
        logger.info("=" * 60)
        logger.info("DataJuicer Job Started")
        logger.info("=" * 60)
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Job Directory: {job_dir}")
        logger.info(f"Work Directory: {self.work_dir}")
        logger.info(f"Event Logs: {job_summary['event_log_file']}")
        logger.info(f"Checkpoints: {job_checkpoint_dir}")
        if event_log_dir:
            logger.info(f"Event Log Storage: {event_log_dir} (configured)")
        if checkpoint_dir:
            logger.info(f"Checkpoint Storage: {checkpoint_dir} (configured)")
        logger.info("=" * 60)
        logger.info("To resume this job later, use:")
        logger.info(f"  {job_summary['resumption_command']}")
        logger.info("=" * 60)

    def _update_job_summary(self, status: str, end_time: Optional[float] = None, error_message: Optional[str] = None):
        """Update job summary with completion status."""
        job_id = self.event_logger.job_id
        job_dir = os.path.join(self.work_dir, job_id)
        summary_file = os.path.join(job_dir, "job_summary.json")

        if not os.path.exists(summary_file):
            return

        with open(summary_file, "r") as f:
            job_summary = json.load(f)

        job_summary.update(
            {
                "status": status,
                "end_time": end_time or time.time(),
                "duration": (end_time or time.time()) - job_summary.get("start_time", time.time()),
                "error_message": error_message,
            }
        )

        with open(summary_file, "w") as f:
            json.dump(job_summary, f, indent=2, default=str)

        # Display completion info
        if status == "completed":
            logger.info("=" * 60)
            logger.info("DataJuicer Job Completed Successfully")
            logger.info(f"Duration: {job_summary['duration']:.2f} seconds")
            logger.info("=" * 60)
        elif status == "failed":
            logger.error("=" * 60)
            logger.error("DataJuicer Job Failed")
            logger.error(f"Error: {error_message}")
            logger.error(f"Duration: {job_summary['duration']:.2f} seconds")
            logger.error("=" * 60)
            logger.error("To resume this job, use:")
            logger.error(f"  {job_summary['resumption_command']}")
            logger.error("=" * 60)

    def _load_job_summary(self) -> Optional[Dict[str, Any]]:
        """Load job summary if it exists."""
        job_id = getattr(self.cfg, "job_id", None)
        if not job_id:
            return None

        job_dir = os.path.join(self.work_dir, job_id)
        summary_file = os.path.join(job_dir, "job_summary.json")

        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                return json.load(f)
        return None

    def _validate_job_resumption(self, job_id: str) -> bool:
        """Validate that the provided job_id matches existing work directory."""
        job_summary = self._load_job_summary()
        if not job_summary:
            logger.warning(f"No job summary found for job_id: {job_id}")
            return False

        if job_summary.get("job_id") != job_id:
            logger.error(f"Job ID mismatch: provided '{job_id}' but found '{job_summary.get('job_id')}'")
            return False

        logger.info(f"Resuming job: {job_id}")
        logger.info(f"Job directory: {job_summary.get('job_dir', 'unknown')}")
        logger.info(f"Previous status: {job_summary.get('status', 'unknown')}")
        if job_summary.get("error_message"):
            logger.warning(f"Previous error: {job_summary['error_message']}")
        return True

    def _get_config_name(self) -> str:
        """Extract a meaningful name from config file or project name."""
        # Try to get config file name first
        config_file = getattr(self.cfg, "config", None)
        if config_file:
            # Extract filename without extension and path
            config_name = os.path.splitext(os.path.basename(config_file))[0]
            # Clean up the name (remove special chars, limit length)
            config_name = re.sub(r"[^a-zA-Z0-9_-]", "_", config_name)
            config_name = config_name[:20]  # Limit length
            if config_name:
                return config_name

        # Fall back to project name
        project_name = getattr(self.cfg, "project_name", "dj")
        # Clean up project name
        project_name = re.sub(r"[^a-zA-Z0-9_-]", "_", project_name)
        project_name = project_name[:15]  # Limit length

        return project_name

    def _log_event(self, event_type: EventType, message: str, **kwargs):
        """Log an event if event logging is enabled."""
        if self.event_logger is None:
            return

        event = Event(event_type=event_type, timestamp=time.time(), message=message, **kwargs)
        self.event_logger.log_event(event)

    # Add new logging methods for job, partition, and op events
    def log_job_start(self, config, total_partitions):
        self._log_event(EventType.JOB_START, "Job started", config=config, total_partitions=total_partitions)

    def log_job_complete(self, status, duration):
        self._log_event(EventType.JOB_COMPLETE, "Job completed", status=status, duration=duration)
        self._update_job_summary("completed", error_message=None)

    def log_job_failed(self, error_message, duration):
        self._log_event(
            EventType.JOB_FAILED, "Job failed", status="failed", error_message=error_message, duration=duration
        )
        self._update_job_summary("failed", error_message=error_message)

    def log_partition_start(self, partition_id, partition_meta):
        self._log_event(
            EventType.PARTITION_START,
            f"Partition {partition_id} started",
            partition_id=partition_id,
            partition_meta=partition_meta,
        )

    def log_partition_complete(self, partition_id, duration, output_path):
        self._log_event(
            EventType.PARTITION_COMPLETE,
            f"Partition {partition_id} completed",
            partition_id=partition_id,
            duration=duration,
            output_path=output_path,
            status="success",
        )

    def log_partition_failed(self, partition_id, error_message, retry_count):
        self._log_event(
            EventType.PARTITION_FAILED,
            f"Partition {partition_id} failed",
            partition_id=partition_id,
            error_message=error_message,
            retry_count=retry_count,
            status="failed",
        )

    def log_op_start(self, partition_id, operation_name, operation_idx, op_args):
        self._log_event(
            EventType.OP_START,
            f"Op {operation_name} (idx {operation_idx}) started on partition {partition_id}",
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            op_args=op_args,
        )

    def log_op_complete(
        self, partition_id, operation_name, operation_idx, duration, checkpoint_path, input_rows, output_rows
    ):
        self._log_event(
            EventType.OP_COMPLETE,
            f"Op {operation_name} (idx {operation_idx}) completed on partition {partition_id}",
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            duration=duration,
            checkpoint_path=checkpoint_path,
            input_rows=input_rows,
            output_rows=output_rows,
            status="success",
        )

    def log_op_failed(self, partition_id, operation_name, operation_idx, error_message, retry_count):
        self._log_event(
            EventType.OP_FAILED,
            f"Op {operation_name} (idx {operation_idx}) failed on partition {partition_id}",
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            error_message=error_message,
            retry_count=retry_count,
            status="failed",
        )

    def log_checkpoint_save(self, partition_id, operation_name, operation_idx, checkpoint_path):
        self._log_event(
            EventType.CHECKPOINT_SAVE,
            f"Checkpoint saved for op {operation_name} (idx {operation_idx}) on partition {partition_id}",
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            checkpoint_path=checkpoint_path,
        )

    def log_checkpoint_load(self, partition_id, operation_name, operation_idx, checkpoint_path):
        self._log_event(
            EventType.CHECKPOINT_LOAD,
            f"Checkpoint loaded for op {operation_name} (idx {operation_idx}) on partition {partition_id}",
            partition_id=partition_id,
            operation_name=operation_name,
            operation_idx=operation_idx,
            checkpoint_path=checkpoint_path,
        )

    def get_events(self, **kwargs) -> List[Event]:
        """Get events with optional filtering."""
        if self.event_logger is None:
            return []
        return self.event_logger.get_events(**kwargs)

    def get_performance_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary."""
        if self.event_logger is None:
            return {}
        return self.event_logger.get_performance_summary(operation_name)

    def generate_status_report(self) -> str:
        """Generate status report."""
        if self.event_logger is None:
            return "Event logging is disabled."
        return self.event_logger.generate_status_report()

    def monitor_events(self, event_type: Optional[EventType] = None) -> Generator[Event, None, None]:
        """Monitor events in real-time."""
        if self.event_logger is None:
            return
        yield from self.event_logger.monitor_events(event_type)
