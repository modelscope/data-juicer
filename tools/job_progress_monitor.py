#!/usr/bin/env python3
"""
DataJuicer Job Progress Monitor

A utility to monitor and display progress information for DataJuicer jobs.
Shows partition status, operation progress, checkpoints, and overall job metrics.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class JobProgressMonitor:
    """Monitor and display progress for DataJuicer jobs."""

    def __init__(self, job_id: str, base_dir: str = "outputs/partition-checkpoint-eventlog"):
        """
        Initialize the job progress monitor.

        Args:
            job_id: The job ID to monitor
            base_dir: Base directory containing job outputs
        """
        self.job_id = job_id
        self.base_dir = Path(base_dir)
        self.job_dir = self.base_dir / job_id

        if not self.job_dir.exists():
            raise FileNotFoundError(f"Job directory not found: {self.job_dir}")

    def load_job_summary(self) -> Dict[str, Any]:
        """Load job summary information."""
        summary_file = self.job_dir / "job_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                return json.load(f)
        return {}

    def load_dataset_mapping(self) -> Dict[str, Any]:
        """Load dataset mapping information."""
        mapping_file = self.job_dir / "metadata" / "dataset_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                return json.load(f)
        return {}

    def load_event_logs(self) -> List[Dict[str, Any]]:
        """Load and parse event logs."""
        events_file = self.job_dir / "event_logs" / "events.jsonl"
        events = []

        if events_file.exists():
            with open(events_file, "r") as f:
                for line in f:
                    try:
                        events.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        return events

    def get_partition_status(self) -> Dict[int, Dict[str, Any]]:
        """Get current status of all partitions."""
        dataset_mapping = self.load_dataset_mapping()
        events = self.load_event_logs()

        partition_status = {}

        # Initialize from dataset mapping
        if "partitions" in dataset_mapping:
            for partition_info in dataset_mapping["partitions"]:
                partition_id = partition_info["partition_id"]
                partition_status[partition_id] = {
                    "status": partition_info.get("processing_status", "unknown"),
                    "sample_count": partition_info.get("sample_count", 0),
                    "start_time": partition_info.get("processing_start_time"),
                    "end_time": partition_info.get("processing_end_time"),
                    "error_message": partition_info.get("error_message"),
                    "current_op": None,
                    "completed_ops": [],
                    "checkpoints": [],
                }

        # Update from event logs
        for event in events:
            if "partition_id" in event:
                partition_id = event["partition_id"]
                if partition_id not in partition_status:
                    partition_status[partition_id] = {
                        "status": "unknown",
                        "sample_count": 0,
                        "start_time": None,
                        "end_time": None,
                        "error_message": None,
                        "current_op": None,
                        "completed_ops": [],
                        "checkpoints": [],
                    }

                # Track partition start/complete
                if event["event_type"] == "partition_start":
                    partition_status[partition_id]["start_time"] = event["timestamp"]
                    partition_status[partition_id]["status"] = "processing"

                elif event["event_type"] == "partition_complete":
                    partition_status[partition_id]["end_time"] = event["timestamp"]
                    partition_status[partition_id]["status"] = "completed"

                # Track operations
                elif event["event_type"] == "op_start":
                    partition_status[partition_id]["current_op"] = {
                        "name": event.get("operation_name", "Unknown"),
                        "idx": event.get("operation_idx", 0),
                        "start_time": event["timestamp"],
                    }

                elif event["event_type"] == "op_complete":
                    op_info = {
                        "name": event.get("operation_name", "Unknown"),
                        "idx": event.get("operation_idx", 0),
                        "duration": event.get("duration", 0),
                        "input_rows": event.get("input_rows", 0),
                        "output_rows": event.get("output_rows", 0),
                        "throughput": event.get("performance_metrics", {}).get("throughput", 0),
                        "reduction_ratio": event.get("performance_metrics", {}).get("reduction_ratio", 0),
                    }
                    partition_status[partition_id]["completed_ops"].append(op_info)
                    partition_status[partition_id]["current_op"] = None

                # Track checkpoints
                elif event["event_type"] == "checkpoint_save":
                    checkpoint_info = {
                        "operation_name": event.get("operation_name", "Unknown"),
                        "operation_idx": event.get("operation_idx", 0),
                        "checkpoint_path": event.get("checkpoint_path", ""),
                        "timestamp": event["timestamp"],
                    }
                    partition_status[partition_id]["checkpoints"].append(checkpoint_info)

        return partition_status

    def get_operation_pipeline(self) -> List[Dict[str, Any]]:
        """Get the operation pipeline from config."""
        config_file = self.job_dir / "partition-checkpoint-eventlog.yaml"
        if not config_file.exists():
            return []

        # Try to find process section in config
        with open(config_file, "r") as f:
            content = f.read()

        # Simple parsing for process section
        operations = []
        lines = content.split("\n")
        in_process = False

        for line in lines:
            if line.strip().startswith("process:"):
                in_process = True
                continue
            elif in_process and line.strip().startswith("-"):
                # Extract operation name
                op_line = line.strip()
                if ":" in op_line:
                    op_name = op_line.split(":")[0].replace("- ", "").strip()
                    operations.append({"name": op_name, "config": {}})

        return operations

    def calculate_overall_progress(self) -> Dict[str, Any]:
        """Calculate overall job progress."""
        partition_status = self.get_partition_status()
        job_summary = self.load_job_summary()

        total_partitions = len(partition_status)
        completed_partitions = sum(1 for p in partition_status.values() if p["status"] == "completed")
        processing_partitions = sum(1 for p in partition_status.values() if p["status"] == "processing")
        failed_partitions = sum(1 for p in partition_status.values() if p["status"] == "failed")

        # Calculate total samples
        total_samples = sum(p.get("sample_count", 0) for p in partition_status.values())
        processed_samples = sum(
            p.get("sample_count", 0) for p in partition_status.values() if p["status"] == "completed"
        )

        # Calculate progress percentage
        progress_percentage = (completed_partitions / total_partitions * 100) if total_partitions > 0 else 0

        # Calculate estimated time remaining
        estimated_remaining = None
        if job_summary and "start_time" in job_summary and completed_partitions > 0:
            elapsed_time = time.time() - job_summary["start_time"]
            if completed_partitions > 0:
                avg_time_per_partition = elapsed_time / completed_partitions
                remaining_partitions = total_partitions - completed_partitions
                estimated_remaining = avg_time_per_partition * remaining_partitions

        return {
            "total_partitions": total_partitions,
            "completed_partitions": completed_partitions,
            "processing_partitions": processing_partitions,
            "failed_partitions": failed_partitions,
            "progress_percentage": progress_percentage,
            "total_samples": total_samples,
            "processed_samples": processed_samples,
            "estimated_remaining_seconds": estimated_remaining,
            "job_status": job_summary.get("status", "unknown"),
        }

    def display_progress(self, detailed: bool = False):
        """Display job progress information."""
        print(f"\n{'='*80}")
        print(f"DataJuicer Job Progress Monitor")
        print(f"Job ID: {self.job_id}")
        print(f"{'='*80}")

        # Load data
        job_summary = self.load_job_summary()
        dataset_mapping = self.load_dataset_mapping()
        partition_status = self.get_partition_status()
        overall_progress = self.calculate_overall_progress()

        # Job overview
        print(f"\n📊 JOB OVERVIEW")
        print(f"   Status: {overall_progress['job_status'].upper()}")
        print(f"   Dataset: {dataset_mapping.get('original_dataset_path', 'Unknown')}")
        print(f"   Total Samples: {dataset_mapping.get('original_dataset_size', 0):,}")
        print(f"   Partition Size: {dataset_mapping.get('partition_size', 0):,} samples")

        if job_summary.get("start_time"):
            start_time = datetime.fromtimestamp(job_summary["start_time"])
            print(f"   Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if job_summary.get("duration"):
            print(f"   Duration: {job_summary['duration']:.1f} seconds")

        # Overall progress
        print(f"\n🎯 OVERALL PROGRESS")
        print(
            f"   Progress: {overall_progress['progress_percentage']:.1f}% "
            f"({overall_progress['completed_partitions']}/{overall_progress['total_partitions']} partitions)"
        )
        print(
            f"   Status: {overall_progress['completed_partitions']} completed, "
            f"{overall_progress['processing_partitions']} processing, "
            f"{overall_progress['failed_partitions']} failed"
        )
        print(f"   Samples: {overall_progress['processed_samples']:,}/{overall_progress['total_samples']:,}")

        if overall_progress["estimated_remaining_seconds"]:
            remaining_minutes = overall_progress["estimated_remaining_seconds"] / 60
            print(f"   Estimated Time Remaining: {remaining_minutes:.1f} minutes")

        # Partition status
        print(f"\n📦 PARTITION STATUS")
        for partition_id in sorted(partition_status.keys()):
            partition = partition_status[partition_id]
            status_icon = {"completed": "✅", "processing": "🔄", "failed": "❌", "unknown": "❓"}.get(
                partition["status"], "❓"
            )

            print(f"   Partition {partition_id:2d}: {status_icon} {partition['status'].upper()}")
            print(f"              Samples: {partition['sample_count']:,}")

            if partition["current_op"]:
                print(f"              Current: {partition['current_op']['name']} (op {partition['current_op']['idx']})")

            if partition["completed_ops"]:
                print(f"              Completed: {len(partition['completed_ops'])} operations")

            if partition["checkpoints"]:
                print(f"              Checkpoints: {len(partition['checkpoints'])} saved")

        if detailed:
            # Detailed operation information
            print(f"\n🔧 OPERATION DETAILS")
            for partition_id in sorted(partition_status.keys()):
                partition = partition_status[partition_id]
                if partition["completed_ops"]:
                    print(f"\n   Partition {partition_id}:")
                    for op in partition["completed_ops"]:
                        reduction = op.get("reduction_ratio", 0) * 100
                        print(
                            f"     {op['name']:25s} | "
                            f"Duration: {op['duration']:6.1f}s | "
                            f"Throughput: {op['throughput']:6.0f} rows/s | "
                            f"Reduction: {reduction:5.2f}%"
                        )

        # Checkpoint information
        print(f"\n💾 CHECKPOINT SUMMARY")
        total_checkpoints = sum(len(p["checkpoints"]) for p in partition_status.values())
        print(f"   Total Checkpoints: {total_checkpoints}")

        if detailed:
            for partition_id in sorted(partition_status.keys()):
                partition = partition_status[partition_id]
                if partition["checkpoints"]:
                    print(f"\n   Partition {partition_id} checkpoints:")
                    for checkpoint in partition["checkpoints"]:
                        checkpoint_time = datetime.fromtimestamp(checkpoint["timestamp"])
                        print(
                            f"     {checkpoint['operation_name']} (op {checkpoint['operation_idx']}) - "
                            f"{checkpoint_time.strftime('%H:%M:%S')}"
                        )

        print(f"\n{'='*80}")

    def get_progress_data(self) -> Dict[str, Any]:
        """Get progress data as a dictionary for programmatic use."""
        job_summary = self.load_job_summary()
        dataset_mapping = self.load_dataset_mapping()
        partition_status = self.get_partition_status()
        overall_progress = self.calculate_overall_progress()

        return {
            "job_id": self.job_id,
            "job_summary": job_summary,
            "dataset_mapping": dataset_mapping,
            "partition_status": partition_status,
            "overall_progress": overall_progress,
        }


def show_job_progress(
    job_id: str, base_dir: str = "outputs/partition-checkpoint-eventlog", detailed: bool = False
) -> Dict[str, Any]:
    """
    Utility function to show job progress.

    Args:
        job_id: The job ID to monitor
        base_dir: Base directory containing job outputs
        detailed: Whether to show detailed operation information

    Returns:
        Dictionary containing all progress data

    Example:
        >>> show_job_progress("20250728_233517_510abf")
        >>> show_job_progress("20250728_233517_510abf", detailed=True)
    """
    monitor = JobProgressMonitor(job_id, base_dir)
    monitor.display_progress(detailed)
    return monitor.get_progress_data()


def main():
    """Main entry point for the job progress monitor."""
    parser = argparse.ArgumentParser(description="Monitor DataJuicer job progress")
    parser.add_argument("job_id", help="Job ID to monitor")
    parser.add_argument(
        "--base-dir", default="outputs/partition-checkpoint-eventlog", help="Base directory containing job outputs"
    )
    parser.add_argument("--detailed", action="store_true", help="Show detailed operation information")
    parser.add_argument("--watch", action="store_true", help="Watch mode - continuously update progress")
    parser.add_argument("--interval", type=int, default=10, help="Update interval in seconds for watch mode")

    args = parser.parse_args()

    try:
        monitor = JobProgressMonitor(args.job_id, args.base_dir)

        if args.watch:
            print(f"Watching job {args.job_id} (press Ctrl+C to stop)...")
            try:
                while True:
                    os.system("clear" if os.name == "posix" else "cls")
                    monitor.display_progress(args.detailed)
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped watching.")
        else:
            monitor.display_progress(args.detailed)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
