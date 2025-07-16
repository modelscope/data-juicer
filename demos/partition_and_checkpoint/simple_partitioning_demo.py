#!/usr/bin/env python3
"""
Simple Partitioning and Checkpointing Demo

This script demonstrates the enhanced features of the original PartitionedRayExecutor:
1. Partitioning/chunking of datasets for fault tolerance
2. Checkpointing support for intermediate data (using Parquet)
3. Event logging system to track partitions and operations
4. Recovery mechanisms for failed partitions
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor


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


def demonstrate_status_monitoring(executor: PartitionedRayExecutor):
    """Demonstrate status monitoring capabilities."""
    print("\n=== Status Monitoring Demonstration ===")
    
    # Get current status
    status = executor.get_status_summary()
    print(f"Current Status:")
    print(f"  Total Partitions: {status['total_partitions']}")
    print(f"  Completed: {status['completed_partitions']}")
    print(f"  Failed: {status['failed_partitions']}")
    print(f"  Processing: {status['processing_partitions']}")
    print(f"  Success Rate: {status['success_rate']:.1%}")
    print(f"  Checkpoints Created: {status['checkpoints_created']}")
    print(f"  Work Directory: {status['work_directory']}")


def demonstrate_event_logging(executor: PartitionedRayExecutor):
    """Demonstrate event logging capabilities."""
    print("\n=== Event Logging Demonstration ===")
    
    # Get all events
    events = executor.get_events()
    
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


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Simple Partitioning and Checkpointing Demo")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--dataset", type=str, help="Dataset file path")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples for synthetic dataset")
    parser.add_argument("--analyze", action="store_true", help="Analyze work directory after processing")
    args = parser.parse_args()
    
    print("🚀 Simple Partitioning and Checkpointing Demo")
    print("=" * 80)
    
    # Create sample dataset if not provided
    if not args.dataset:
        dataset_path = "demos/data/simple_demo_dataset.jsonl"
        create_sample_dataset(dataset_path, args.samples)
    else:
        dataset_path = args.dataset
    
    # Load configuration
    if args.config:
        cfg = init_configs(args.config)
    else:
        # Create default configuration
        cfg = init_configs()
        
        # Set configuration for simple demo
        cfg.project_name = 'simple-partitioning-demo'
        cfg.dataset_path = dataset_path
        cfg.export_path = 'demos/output/simple_result.jsonl'
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
        cfg.work_dir = './demos/work_dir_simple'
    
    print(f"Configuration:")
    print(f"  Dataset: {cfg.dataset_path}")
    print(f"  Partition Size: {getattr(cfg, 'partition_size', 'N/A')} samples")
    print(f"  Storage Format: {getattr(cfg, 'storage_format', 'N/A')}")
    print(f"  Fault Tolerance: {getattr(cfg, 'enable_fault_tolerance', 'N/A')}")
    print(f"  Intermediate Data: {getattr(cfg, 'preserve_intermediate_data', 'N/A')}")
    print(f"  Work Directory: {getattr(cfg, 'work_dir', 'N/A')}")
    
    # Create executor
    executor = PartitionedRayExecutor(cfg)
    
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
        
        # Demonstrate features
        demonstrate_status_monitoring(executor)
        demonstrate_event_logging(executor)
        
        # Analyze work directory if requested
        if args.analyze:
            analyze_work_directory(cfg.work_dir)
        
        print(f"\n=== Demo Completed Successfully ===")
        print(f"Results saved to: {cfg.export_path}")
        print(f"Work directory: {cfg.work_dir}")
        print(f"Check the work directory for detailed logs and metadata")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        raise


if __name__ == "__main__":
    main() 