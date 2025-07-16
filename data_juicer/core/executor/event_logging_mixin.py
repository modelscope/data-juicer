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
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from loguru import logger


class EventType(Enum):
    """Types of events that can be logged."""

    OPERATION_START = "operation_start"
    OPERATION_COMPLETE = "operation_complete"
    OPERATION_ERROR = "operation_error"
    PARTITION_START = "partition_start"
    PARTITION_COMPLETE = "partition_complete"
    PARTITION_ERROR = "partition_error"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    DATASET_LOAD = "dataset_load"
    DATASET_SAVE = "dataset_save"
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
    partition_id: Optional[int] = None
    operation_name: Optional[str] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    resource_usage: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class EventLogger:
    """Event logging system with real-time capabilities."""

    def __init__(self, log_dir: str, max_log_size_mb: int = 100, backup_count: int = 5):
        """Initialize the event logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_log_size_mb = max_log_size_mb
        self.backup_count = backup_count

        # Event storage
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events in memory
        self.event_lock = threading.Lock()

        # Performance tracking
        self.performance_metrics = defaultdict(list)
        self.resource_usage = defaultdict(list)

        # Setup file logging
        self._setup_file_logging()

        # Start background cleanup thread
        self._start_cleanup_thread()

    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = self.log_dir / "events.log"

        # Configure loguru for file logging
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            level="INFO",
            rotation=f"{self.max_log_size_mb} MB",
            retention=f"{self.backup_count} files",
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
        """Log an event."""
        with self.event_lock:
            self.events.append(event)

            # Log to file
            log_message = self._format_event_for_logging(event)
            logger.info(log_message)

            # Track performance metrics
            if event.performance_metrics:
                self.performance_metrics[event.operation_name or "unknown"].append(event.performance_metrics)

            # Track resource usage
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


class EventLoggingMixin:
    """Mixin to add event logging capabilities to any executor."""

    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        super().__init__(*args, **kwargs)

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

        # Setup event logger
        log_dir = os.path.join(self.work_dir, "event_logs")
        max_log_size = event_config.get("max_log_size_mb", 100)
        backup_count = event_config.get("backup_count", 5)

        self.event_logger = EventLogger(log_dir, max_log_size, backup_count)

        # Log initialization
        self._log_event(EventType.INFO, f"Event logging initialized for {self.executor_type} executor")

    def _log_event(self, event_type: EventType, message: str, **kwargs):
        """Log an event if event logging is enabled."""
        if self.event_logger is None:
            return

        event = Event(event_type=event_type, timestamp=time.time(), message=message, **kwargs)
        self.event_logger.log_event(event)

    def _log_operation_start(self, operation_name: str, partition_id: Optional[int] = None, **kwargs):
        """Log the start of an operation."""
        self._log_event(
            EventType.OPERATION_START,
            f"Starting operation: {operation_name}",
            operation_name=operation_name,
            partition_id=partition_id,
            **kwargs,
        )

    def _log_operation_complete(
        self, operation_name: str, duration: float, partition_id: Optional[int] = None, **kwargs
    ):
        """Log the completion of an operation."""
        self._log_event(
            EventType.OPERATION_COMPLETE,
            f"Completed operation: {operation_name} in {duration:.3f}s",
            operation_name=operation_name,
            duration=duration,
            partition_id=partition_id,
            **kwargs,
        )

    def _log_operation_error(self, operation_name: str, error: Exception, partition_id: Optional[int] = None, **kwargs):
        """Log an operation error."""
        import traceback

        self._log_event(
            EventType.OPERATION_ERROR,
            f"Error in operation: {operation_name} - {str(error)}",
            operation_name=operation_name,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            partition_id=partition_id,
            **kwargs,
        )

    def _log_performance_metric(self, operation_name: str, duration: float, throughput: float, **kwargs):
        """Log a performance metric."""
        self._log_event(
            EventType.PERFORMANCE_METRIC,
            f"Performance: {operation_name} - {duration:.3f}s, {throughput:.1f} samples/s",
            operation_name=operation_name,
            duration=duration,
            performance_metrics={"duration": duration, "throughput": throughput},
            **kwargs,
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
