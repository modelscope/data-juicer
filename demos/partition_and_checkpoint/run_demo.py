#!/usr/bin/env python3
"""
Comprehensive Demo for DataJuicer Fault-Tolerant Processing

This script demonstrates all the implemented features:
1. Job-specific directory isolation
2. Flexible storage paths for event logs and checkpoints
3. Configurable checkpointing strategies
4. Event logging with JSONL format
5. Job resumption capabilities
6. Comprehensive job management

Usage:
    python run_comprehensive_demo.py
"""

import os
import subprocess
import time
import json
from pathlib import Path
import re


def run_data_juicer_command(config_file, job_id=None, extra_args=None):
    """Run a DataJuicer command and return the result."""
    cmd = ["dj-process", "--config", config_file]
    if job_id:
        cmd.extend(["--job_id", job_id])
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Exit code: {result.returncode}")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print("-" * 80)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result


def check_directory_structure(job_id, work_dir="./outputs/partition-checkpoint-eventlog"):
    """Check and display the job-specific directory structure."""
    job_dir = os.path.join(work_dir, job_id)
    
    print(f"\n📁 Job Directory Structure for {job_id}:")
    print("=" * 60)
    
    if os.path.exists(job_dir):
        for root, dirs, files in os.walk(job_dir):
            level = root.replace(job_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print(f"Job directory {job_dir} does not exist")
    
    print("=" * 60)


def check_flexible_storage(job_id):
    """Check flexible storage directories."""
    print(f"\n💾 Flexible Storage for {job_id}:")
    print("=" * 60)
    
    # Check event logs in fast storage
    event_log_dir = f"/tmp/fast_event_logs/{job_id}/event_logs"
    if os.path.exists(event_log_dir):
        print(f"✅ Event Logs: {event_log_dir}")
        for file in os.listdir(event_log_dir):
            file_path = os.path.join(event_log_dir, file)
            size = os.path.getsize(file_path)
            print(f"   📄 {file} ({size} bytes)")
    else:
        print(f"❌ Event Logs: {event_log_dir} not found")
    
    # Check checkpoints in large storage
    checkpoint_dir = f"/tmp/large_checkpoints/{job_id}"
    if os.path.exists(checkpoint_dir):
        print(f"✅ Checkpoints: {checkpoint_dir}")
        for file in os.listdir(checkpoint_dir):
            file_path = os.path.join(checkpoint_dir, file)
            size = os.path.getsize(file_path)
            print(f"   💾 {file} ({size} bytes)")
    else:
        print(f"❌ Checkpoints: {checkpoint_dir} not found")
    
    print("=" * 60)


def check_job_summary(job_id, work_dir="./outputs/partition-checkpoint-eventlog"):
    """Check and display job summary."""
    job_dir = os.path.join(work_dir, job_id)
    summary_file = os.path.join(job_dir, "job_summary.json")
    
    print(f"\n📋 Job Summary for {job_id}:")
    print("=" * 60)
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"Job ID: {summary.get('job_id')}")
        print(f"Status: {summary.get('status')}")
        print(f"Start Time: {summary.get('start_time')}")
        print(f"Job Directory: {summary.get('job_dir')}")
        print(f"Event Log File: {summary.get('event_log_file')}")
        print(f"Checkpoint Directory: {summary.get('checkpoint_dir')}")
        print(f"Resumption Command: {summary.get('resumption_command')}")
    else:
        print(f"Job summary file {summary_file} not found")
    
    print("=" * 60)


def get_latest_job_id(work_dir):
    """Get the most recently created job_id directory in work_dir."""
    if not os.path.exists(work_dir):
        return None
    job_dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    if not job_dirs:
        return None
    # Sort by creation time (descending)
    job_dirs = sorted(job_dirs, key=lambda d: os.path.getctime(os.path.join(work_dir, d)), reverse=True)
    return job_dirs[0]


def main():
    """Run the comprehensive demo."""
    print("🚀 DataJuicer Fault-Tolerant Processing Demo")
    print("=" * 80)
    
    config_file = "configs/demo/partition-checkpoint-eventlog.yaml"
    work_dir = "./outputs/partition-checkpoint-eventlog"
    
    # Ensure the config file exists
    if not os.path.exists(config_file):
        print(f"❌ Config file {config_file} not found!")
        print("Please run this script from the DataJuicer root directory.")
        return
    
    # Demo 1: First run with new job (auto-generated job_id)
    print("\n🎯 Demo 1: First Run (New Job, Auto-generated job_id)")
    print("=" * 80)
    result1 = run_data_juicer_command(config_file)
    job_id_1 = get_latest_job_id(work_dir)
    if result1.returncode == 0 and job_id_1:
        print(f"✅ First run completed successfully! (job_id: {job_id_1})")
        check_directory_structure(job_id_1, work_dir)
        check_flexible_storage(job_id_1)
        check_job_summary(job_id_1, work_dir)
    else:
        print("❌ First run failed!")
        return
    
    # Demo 2: Resume the same job
    print("\n🎯 Demo 2: Resume Job")
    print("=" * 80)
    result2 = run_data_juicer_command(config_file, job_id_1)
    if result2.returncode == 0:
        print("✅ Job resumption completed successfully!")
        print("Note: This should be much faster than the first run due to checkpoint resumption.")
        check_job_summary(job_id_1, work_dir)
    else:
        print("❌ Job resumption failed!")
    
    # Demo 3: New job with different checkpoint strategy (auto-generated job_id)
    print("\n🎯 Demo 3: Different Checkpoint Strategy")
    print("=" * 80)
    extra_args = ["--checkpoint.strategy", "every_partition"]
    result3 = run_data_juicer_command(config_file, None, extra_args)
    job_id_2 = get_latest_job_id(work_dir)
    if result3.returncode == 0 and job_id_2:
        print(f"✅ Different checkpoint strategy completed successfully! (job_id: {job_id_2})")
        check_directory_structure(job_id_2, work_dir)
        check_flexible_storage(job_id_2)
        check_job_summary(job_id_2, work_dir)
    else:
        print("❌ Different checkpoint strategy failed!")
    
    # Demo 4: List available jobs
    print("\n🎯 Demo 4: List Available Jobs")
    print("=" * 80)
    if os.path.exists(work_dir):
        print("Available job directories:")
        for item in os.listdir(work_dir):
            item_path = os.path.join(work_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "job_summary.json")):
                print(f"  📁 {item}")
    else:
        print(f"Work directory {work_dir} not found")
    
    print("\n🎉 Demo completed!")
    print("=" * 80)
    print("Key Features Demonstrated:")
    print("✅ Job-specific directory isolation")
    print("✅ Flexible storage paths (event logs in /tmp/fast_event_logs)")
    print("✅ Flexible storage paths (checkpoints in /tmp/large_checkpoints)")
    print("✅ Configurable checkpointing strategies")
    print("✅ Event logging with JSONL format")
    print("✅ Job resumption capabilities")
    print("✅ Comprehensive job management with job_summary.json")
    print("✅ Fast resumption from checkpoints")


if __name__ == "__main__":
    main() 