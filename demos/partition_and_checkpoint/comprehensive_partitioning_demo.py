#!/usr/bin/env python3
"""
Comprehensive Partitioning and Checkpointing Demo

This script demonstrates:
1. Partitioning/chunking of datasets for fault tolerance
2. Checkpointing support for intermediate data (using Parquet)
3. Event logging system to track partitions and operations
4. Recovery mechanisms for failed partitions
5. Real-time monitoring and status reporting
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import signal
import sys

from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor


class ProcessingMonitor:
    """Real-time monitoring of processing status."""
    
    def __init__(self, executor: PartitionedRayExecutor, refresh_interval: int = 5):
        self.executor = executor
        self.refresh_interval = refresh_interval
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start real-time monitoring in a separate thread."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._print_status()
                time.sleep(self.refresh_interval)
            except KeyboardInterrupt:
                break
    
    def _print_status(self):
        """Print current processing status."""
        status = self.executor.get_status_summary()
        
        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("COMPREHENSIVE PARTITIONING AND CHECKPOINTING DEMO")
        print("=" * 80)
        print(f"Work Directory: {status['work_directory']}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Partition status
        print("PARTITION STATUS:")
        print(f"  Total Partitions: {status['total_partitions']}")
        print(f"  Completed: {status['completed_partitions']}")
        print(f"  Failed: {status['failed_partitions']}")
        print(f"  Processing: {status['processing_partitions']}")
        print(f"  Success Rate: {status['success_rate']:.1%}")
        print()
        
        # Checkpoint status
        print("CHECKPOINT STATUS:")
        print(f"  Checkpoints Created: {status['checkpoints_created']}")
        print()
        
        # Recent events
        self._print_recent_events()
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def _print_recent_events(self):
        """Print recent processing events."""
        try:
            events = self.executor.get_events()
            recent_events = events[-10:]  # Last 10 events
            
            print("RECENT EVENTS:")
            for event in recent_events:
                timestamp = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
                event_type = event.event_type.replace('_', ' ').title()
                
                if event.partition_id is not None:
                    print(f"  [{timestamp}] {event_type} - Partition {event.partition_id}")
                elif event.operation_name:
                    print(f"  [{timestamp}] {event_type} - {event.operation_name}")
                else:
                    print(f"  [{timestamp}] {event_type}")
            
            print()
        except Exception as e:
            print(f"Error reading events: {e}")
            print()


def create_sample_dataset(output_path: str, num_samples: int = 10000):
    """Create a sample dataset for demonstration."""
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Creating sample dataset with {num_samples} samples...")
    
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            sample = {
                'id': i,
                'text': f'This is sample text number {i} for comprehensive partitioning and checkpointing demonstration. '
                       f'It contains various types of content including technical documentation, creative writing, '
                       f'and educational materials to test the robustness of our processing pipeline.',
                'category': f'category_{i % 20}',
                'score': i * 0.01,
                'metadata': {
                    'created_at': time.time(),
                    'version': '1.0',
                    'tags': [f'tag_{j}' for j in range(i % 5 + 1)],
                    'features': {
                        'length': len(f'This is sample text number {i}'),
                        'complexity': i % 10,
                        'quality': i % 100 / 100.0
                    }
                }
            }
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created sample dataset: {output_path} ({num_samples} samples)")


def analyze_work_directory(work_dir: str):
    """Analyze the work directory structure and contents."""
    print(f"\n=== Work Directory Analysis: {work_dir} ===")
    
    if not os.path.exists(work_dir):
        print("Work directory does not exist yet.")
        return
    
    # Directory structure
    print("\nDirectory Structure:")
    for root, dirs, files in os.walk(work_dir):
        level = root.replace(work_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size:,} bytes)")
    
    # Metadata analysis
    metadata_dir = os.path.join(work_dir, "metadata")
    if os.path.exists(metadata_dir):
        print(f"\nMetadata Files:")
        for file in os.listdir(metadata_dir):
            file_path = os.path.join(metadata_dir, file)
            if file.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"  {file}: {len(json.dumps(data))} characters")
    
    # Checkpoint analysis
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        print(f"\nCheckpoint Analysis:")
        checkpoint_count = 0
        total_checkpoint_size = 0
        
        for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith(('.parquet', '.arrow', '.jsonl')):
                    checkpoint_count += 1
                    file_path = os.path.join(root, file)
                    total_checkpoint_size += os.path.getsize(file_path)
        
        print(f"  Total Checkpoints: {checkpoint_count}")
        print(f"  Total Checkpoint Size: {total_checkpoint_size / (1024*1024):.2f} MB")
    
    # Event log analysis
    log_dir = os.path.join(work_dir, "logs")
    if os.path.exists(log_dir):
        events_file = os.path.join(log_dir, "processing_events.jsonl")
        if os.path.exists(events_file):
            event_count = 0
            with open(events_file, 'r') as f:
                for line in f:
                    event_count += 1
            
            print(f"\nEvent Log Analysis:")
            print(f"  Total Events: {event_count}")
            
            # Event type breakdown
            event_types = {}
            with open(events_file, 'r') as f:
                for line in f:
                    event_data = json.loads(line.strip())
                    event_type = event_data.get('event_type', 'unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1
            
            print(f"  Event Types:")
            for event_type, count in sorted(event_types.items()):
                print(f"    {event_type}: {count}")


def demonstrate_fault_tolerance(executor: PartitionedRayExecutor):
    """Demonstrate fault tolerance features."""
    print("\n=== Fault Tolerance Demonstration ===")
    
    # Show partition status
    status = executor.get_status_summary()
    print(f"Current Status:")
    print(f"  Total Partitions: {status['total_partitions']}")
    print(f"  Completed: {status['completed_partitions']}")
    print(f"  Failed: {status['failed_partitions']}")
    print(f"  Success Rate: {status['success_rate']:.1%}")
    
    # Show failed partitions
    failed_partitions = [p for p in executor.partitions if p.processing_status == "failed"]
    if failed_partitions:
        print(f"\nFailed Partitions:")
        for partition in failed_partitions:
            print(f"  Partition {partition.partition_id}: {partition.error_message}")
        
        # Show recovery options
        print(f"\nRecovery Options:")
        print(f"  - Retry failed partitions (max {executor.max_retries} retries)")
        print(f"  - Use checkpoints to resume from last successful operation")
        print(f"  - Restart from beginning if no checkpoints available")
    
    # Show checkpoint availability
    print(f"\nCheckpoint Availability:")
    for partition in executor.partitions:
        latest_checkpoint = executor.checkpoint_manager.get_latest_checkpoint(partition.partition_id)
        if latest_checkpoint:
            print(f"  Partition {partition.partition_id}: Checkpoint at {latest_checkpoint.operation_name}")
        else:
            print(f"  Partition {partition.partition_id}: No checkpoints available")


def demonstrate_event_logging(executor: PartitionedRayExecutor):
    """Demonstrate event logging capabilities."""
    print("\n=== Event Logging Demonstration ===")
    
    # Get all events
    events = executor.event_logger.get_events()
    
    print(f"Total Events: {len(events)}")
    
    # Event type breakdown
    event_types = {}
    for event in events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    print(f"\nEvent Type Breakdown:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    
    # Show recent events
    print(f"\nRecent Events (last 10):")
    recent_events = events[-10:]
    for event in recent_events:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))
        print(f"  [{timestamp}] {event.event_type}: {event.message}")
    
    # Show partition-specific events
    if executor.partitions:
        partition_id = executor.partitions[0].partition_id
        partition_events = executor.event_logger.get_events(partition_id=partition_id)
        print(f"\nEvents for Partition {partition_id}:")
        for event in partition_events:
            timestamp = time.strftime('%H:%M:%S', time.localtime(event.timestamp))
            print(f"  [{timestamp}] {event.event_type}: {event.message}")


def demonstrate_checkpointing(executor: PartitionedRayExecutor):
    """Demonstrate checkpointing capabilities."""
    print("\n=== Checkpointing Demonstration ===")
    
    checkpoints = executor.checkpoint_manager.checkpoints
    
    print(f"Total Checkpoints: {len(checkpoints)}")
    
    if checkpoints:
        # Group by partition
        partition_checkpoints = {}
        for checkpoint in checkpoints:
            if checkpoint.partition_id not in partition_checkpoints:
                partition_checkpoints[checkpoint.partition_id] = []
            partition_checkpoints[checkpoint.partition_id].append(checkpoint)
        
        print(f"\nCheckpoints by Partition:")
        for partition_id, partition_checkpoints_list in sorted(partition_checkpoints.items()):
            print(f"  Partition {partition_id}: {len(partition_checkpoints_list)} checkpoints")
            
            # Show operation progression
            sorted_checkpoints = sorted(partition_checkpoints_list, key=lambda c: c.operation_idx)
            for checkpoint in sorted_checkpoints:
                print(f"    - Operation {checkpoint.operation_idx}: {checkpoint.operation_name} "
                      f"({checkpoint.sample_count} samples, {checkpoint.file_size_bytes:,} bytes)")
        
        # Show checkpoint metadata
        print(f"\nCheckpoint Metadata Example:")
        example_checkpoint = checkpoints[0]
        print(f"  Operation: {example_checkpoint.operation_name}")
        print(f"  Partition: {example_checkpoint.partition_id}")
        print(f"  Samples: {example_checkpoint.sample_count}")
        print(f"  File Size: {example_checkpoint.file_size_bytes:,} bytes")
        print(f"  Checksum: {example_checkpoint.checksum[:16]}...")
        print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(example_checkpoint.timestamp))}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Comprehensive Partitioning and Checkpointing Demo")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--dataset", type=str, help="Dataset file path")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples for synthetic dataset")
    parser.add_argument("--monitor", action="store_true", help="Enable real-time monitoring")
    parser.add_argument("--analyze", action="store_true", help="Analyze work directory after processing")
    args = parser.parse_args()
    
    print("🚀 Comprehensive Partitioning and Checkpointing Demo")
    print("=" * 80)
    
    # Create sample dataset if not provided
    if not args.dataset:
        dataset_path = "demos/data/comprehensive_demo_dataset.jsonl"
        create_sample_dataset(dataset_path, args.samples)
    else:
        dataset_path = args.dataset
    
    # Load configuration
    if args.config:
        cfg = init_configs(args.config)
    else:
        # Create default configuration
        cfg = init_configs()
        
        # Set configuration for comprehensive demo
        cfg.project_name = 'comprehensive-partitioning-demo'
        cfg.dataset_path = dataset_path
        cfg.export_path = 'demos/output/comprehensive_result.jsonl'
        cfg.executor_type = 'ray_partitioned'
        cfg.ray_address = 'auto'
        
        # Partitioning configuration
        cfg.partition_size = 1000  # Small partitions for demo
        cfg.max_partition_size_mb = 64
        cfg.enable_fault_tolerance = True
        cfg.max_retries = 3
        cfg.preserve_intermediate_data = True  # Enable for demo
        cfg.cleanup_temp_files = False  # Keep files for analysis
        
        # Storage format
        cfg.storage_format = 'parquet'  # Use Parquet for checkpoints
        
        # Processing pipeline
        cfg.process = [
            'whitespace_normalization_mapper',
            'text_length_filter',
            'language_id_score_filter'
        ]
        
        # Work directory
        cfg.work_dir = './demos/work_dir_comprehensive'
    
    print(f"Configuration:")
    print(f"  Dataset: {cfg.dataset_path}")
    print(f"  Partition Size: {getattr(cfg, 'partition_size', 'N/A')} samples")
    print(f"  Storage Format: {getattr(cfg, 'storage_format', 'N/A')}")
    print(f"  Fault Tolerance: {getattr(cfg, 'enable_fault_tolerance', 'N/A')}")
    print(f"  Intermediate Data: {getattr(cfg, 'preserve_intermediate_data', 'N/A')}")
    print(f"  Work Directory: {getattr(cfg, 'work_dir', 'N/A')}")
    
    # Create executor
    executor = PartitionedRayExecutor(cfg)
    
    # Set up monitoring if requested
    monitor = None
    if args.monitor:
        monitor = ProcessingMonitor(executor)
        monitor.start_monitoring()
    
    # Set up signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal. Shutting down gracefully...")
        if monitor:
            monitor.stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run processing
        print(f"\n=== Starting Processing ===")
        start_time = time.time()
        
        result_dataset = executor.run()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n=== Processing Completed ===")
        print(f"Total Processing Time: {processing_time:.2f} seconds")
        print(f"Final Dataset Size: {len(result_dataset)} samples")
        
        # Stop monitoring
        if monitor:
            monitor.stop_monitoring()
        
        # Demonstrate features
        demonstrate_fault_tolerance(executor)
        demonstrate_event_logging(executor)
        demonstrate_checkpointing(executor)
        
        # Analyze work directory if requested
        if args.analyze:
            analyze_work_directory(cfg.work_dir)
        
        print(f"\n=== Demo Completed Successfully ===")
        print(f"Results saved to: {cfg.export_path}")
        print(f"Work directory: {cfg.work_dir}")
        print(f"Check the work directory for detailed logs, checkpoints, and metadata")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        if monitor:
            monitor.stop_monitoring()
        raise


if __name__ == "__main__":
    main() 