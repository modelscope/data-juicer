#!/usr/bin/env python3
"""
Demonstration of Partitioned Ray Executor Data Persistence and Mapping Features

This script demonstrates:
1. How intermediate data is saved to disk
2. How mapping between original dataset and partitions is preserved
3. How to inspect and analyze the mapping information
4. How to recover from failures using the preserved data
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any

from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor


def create_sample_dataset(output_path: str, num_samples: int = 1000):
    """Create a sample dataset for demonstration."""
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i in range(num_samples):
            sample = {
                'id': i,
                'text': f'This is sample text number {i} for demonstration purposes.',
                'category': f'category_{i % 10}',
                'metadata': {
                    'created_at': time.time(),
                    'version': '1.0'
                }
            }
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created sample dataset: {output_path} ({num_samples} samples)")


def inspect_work_directory(work_dir: str):
    """Inspect the work directory structure and contents."""
    print(f"\n=== Work Directory Structure: {work_dir} ===")
    
    if not os.path.exists(work_dir):
        print("Work directory does not exist yet.")
        return
    
    for root, dirs, files in os.walk(work_dir):
        level = root.replace(work_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size} bytes)")


def analyze_dataset_mapping(work_dir: str):
    """Analyze the dataset mapping information."""
    mapping_path = os.path.join(work_dir, "metadata", "dataset_mapping.json")
    
    if not os.path.exists(mapping_path):
        print("Dataset mapping file not found.")
        return
    
    print(f"\n=== Dataset Mapping Analysis ===")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    print(f"Original Dataset: {mapping['original_dataset_path']}")
    print(f"Original Size: {mapping['original_dataset_size']:,} samples")
    print(f"Partition Count: {mapping['partition_count']}")
    print(f"Partition Size: {mapping['partition_size']:,} samples")
    print(f"Created: {time.ctime(mapping['created_timestamp'])}")
    
    print(f"\nPartition Details:")
    for partition in mapping['partitions']:
        status = partition['processing_status']
        status_icon = "✅" if status == "completed" else "⏳" if status == "processing" else "❌"
        print(f"  {status_icon} Partition {partition['partition_id']:3d}: "
              f"{partition['original_start_idx']:6d}-{partition['original_end_idx']:6d} "
              f"({partition['sample_count']:5d} samples, {partition['file_size_bytes']:8d} bytes)")


def analyze_intermediate_data(work_dir: str):
    """Analyze intermediate data if preserved."""
    intermediate_dir = os.path.join(work_dir, "intermediate")
    
    if not os.path.exists(intermediate_dir):
        print("\n=== Intermediate Data Analysis ===")
        print("Intermediate data preservation is disabled.")
        return
    
    print(f"\n=== Intermediate Data Analysis ===")
    
    for partition_dir in sorted(os.listdir(intermediate_dir)):
        partition_path = os.path.join(intermediate_dir, partition_dir)
        if os.path.isdir(partition_path):
            print(f"\n{partition_dir}:")
            for file in sorted(os.listdir(partition_path)):
                file_path = os.path.join(partition_path, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size} bytes)")


def analyze_final_report(work_dir: str):
    """Analyze the final mapping report."""
    report_path = os.path.join(work_dir, "metadata", "final_mapping_report.json")
    
    if not os.path.exists(report_path):
        print("Final mapping report not found.")
        return
    
    print(f"\n=== Final Processing Report ===")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    original = report['original_dataset']
    summary = report['processing_summary']
    
    print(f"Original Dataset: {original['path']}")
    print(f"Original Samples: {original['total_samples']:,}")
    print(f"Partitions: {original['partition_count']}")
    print(f"\nProcessing Summary:")
    print(f"  Total Partitions: {summary['total_partitions']}")
    print(f"  Successful: {summary['successful_partitions']}")
    print(f"  Failed: {summary['failed_partitions']}")
    print(f"  Processed Samples: {summary['total_processed_samples']:,}")
    
    success_rate = (summary['successful_partitions'] / summary['total_partitions']) * 100
    print(f"  Success Rate: {success_rate:.1f}%")


def demonstrate_partitioned_processing():
    """Demonstrate the partitioned processing with data persistence."""
    
    # Create sample dataset
    sample_data_path = "demos/data/sample_dataset.jsonl"
    create_sample_dataset(sample_data_path, num_samples=5000)
    
    # Configuration for demonstration
    config = {
        'project_name': 'partitioned-ray-demo',
        'dataset_path': sample_data_path,
        'export_path': 'demos/output/partitioned_result.jsonl',
        'executor_type': 'ray_partitioned',
        'ray_address': 'auto',
        
        # Partitioning configuration
        'partition_size': 1000,  # Small partitions for demo
        'max_partition_size_mb': 64,
        'enable_fault_tolerance': True,
        'max_retries': 2,
        'preserve_intermediate_data': True,  # Enable for demo
        'cleanup_temp_files': False,  # Keep files for inspection
        
        # Processing pipeline
        'process': [
            'whitespace_normalization_mapper',
            'text_length_filter',
            'language_id_score_filter'
        ],
        
        # Work directory
        'work_dir': './demos/work_dir'
    }
    
    # Initialize configuration
    cfg = init_configs()
    for key, value in config.items():
        setattr(cfg, key, value)
    
    print("=== Partitioned Ray Executor Demonstration ===")
    print(f"Dataset: {sample_data_path}")
    print(f"Partition Size: {config['partition_size']} samples")
    print(f"Intermediate Data: {'Enabled' if config['preserve_intermediate_data'] else 'Disabled'}")
    print(f"Work Directory: {config['work_dir']}")
    
    # Create executor and run processing
    try:
        executor = PartitionedRayExecutor(cfg)
        
        print("\n=== Starting Partitioned Processing ===")
        start_time = time.time()
        
        # Run the processing
        result_dataset = executor.run()
        
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        
        # Analyze results
        inspect_work_directory(config['work_dir'])
        analyze_dataset_mapping(config['work_dir'])
        analyze_intermediate_data(config['work_dir'])
        analyze_final_report(config['work_dir'])
        
        # Get processing statistics
        stats = executor.get_processing_stats()
        print(f"\n=== Processing Statistics ===")
        print(f"Status: {stats['status']}")
        print(f"Progress: {stats['progress']:.1f}%")
        print(f"Successful Partitions: {stats['successful_partitions']}/{stats['total_partitions']}")
        
        # Get partition mapping
        mapping = executor.get_partition_mapping()
        if mapping:
            print(f"\n=== Partition Mapping ===")
            print(f"Original Dataset Size: {mapping.original_dataset_size:,} samples")
            print(f"Partition Count: {mapping.partition_count}")
            print(f"Mapping Version: {mapping.mapping_version}")
        
        print(f"\n=== Demonstration Complete ===")
        print(f"Output: {config['export_path']}")
        print(f"Work Directory: {config['work_dir']}")
        print("\nYou can inspect the work directory to see:")
        print("- Partition files in partitions/")
        print("- Intermediate data in intermediate/ (if enabled)")
        print("- Processing results in results/")
        print("- Mapping metadata in metadata/")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_partitioned_processing() 