# Comprehensive Partitioning and Checkpointing Guide

This guide covers the practical usage of the Data-Juicer partitioning and checkpointing system, providing hands-on examples, troubleshooting, and best practices for building fault-tolerant, scalable, and observable data processing pipelines.

> **📚 For detailed architecture information and visual diagrams, see:**
> - [Partitioning_Checkpointing_EventLogging_Architecture.md](Partitioning_Checkpointing_EventLogging_Architecture.md) - Complete architecture documentation with visual diagrams
> - [Partitioning_Checkpointing_EventLogging_Summary.md](Partitioning_Checkpointing_EventLogging_Summary.md) - Executive overview and quick reference

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Guide](#configuration-guide)
4. [Usage Examples](#usage-examples)
5. [Monitoring and Debugging](#monitoring-and-debugging)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Work Directory Structure](#work-directory-structure)

## Overview

The Data-Juicer partitioning and checkpointing system provides enterprise-grade solutions for processing large datasets:

- **🔧 Fault Tolerance**: Automatic recovery from failures using checkpoints
- **📈 Scalability**: Partition-based processing for datasets of any size
- **👁️ Observability**: Comprehensive event logging and real-time monitoring
- **⚡ Performance**: Optimized storage formats and parallel processing
- **🔄 Flexibility**: Configurable partitioning and checkpointing strategies

### Key Components

- **Partitioning Engine**: Splits large datasets into manageable chunks
- **Checkpoint Manager**: Saves and restores processing state
- **Event Logger**: Tracks all operations and performance metrics
- **Ray Cluster**: Provides distributed processing capabilities
- **Result Merger**: Combines processed partitions into final output

## Quick Start

### 1. Basic Configuration

```yaml
# Basic configuration
project_name: 'my-partitioned-project'
dataset_path: 'data/large-dataset.jsonl'
export_path: 'outputs/processed-dataset.jsonl'
executor_type: 'ray_partitioned'

# Ray configuration
ray_address: 'auto'

# Partitioning configuration
partition_size: 10000
max_partition_size_mb: 128
enable_fault_tolerance: true
max_retries: 3

# Storage configuration
storage_format: 'parquet'
preserve_intermediate_data: true

# Event logging
event_logging:
  enabled: true
  log_level: 'INFO'
  max_log_size_mb: 100
  backup_count: 5

# Processing pipeline
process:
  - whitespace_normalization_mapper:
  - text_length_filter:
      min_len: 50
      max_len: 2000
  - language_id_score_filter:
      lang: 'en'
      min_score: 0.8
```

### 2. Basic Usage

```python
from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor

# Load configuration
cfg = init_configs()

# Create executor
executor = PartitionedRayExecutor(cfg)

# Run processing
result = executor.run()

# Get events and performance data
events = executor.get_events()
perf_summary = executor.get_performance_summary()
print(f"Logged {len(events)} events")
print(f"Performance: {perf_summary}")
```

## Configuration Guide

### Partitioning Configuration

```yaml
# Partitioning settings
partition_size: 10000              # Samples per partition
max_partition_size_mb: 128         # Maximum partition file size
enable_fault_tolerance: true       # Enable fault tolerance
max_retries: 3                     # Maximum retry attempts
```

**Partitioning Strategies:**
- **Sample-based**: Control number of samples per partition
- **Size-based**: Control partition file size
- **Adaptive**: Automatic size calculation based on dataset characteristics

### Checkpointing Configuration

```yaml
# Checkpointing settings
preserve_intermediate_data: true
storage_format: 'parquet'          # parquet, arrow, jsonl

checkpointing:
  enabled: true
  storage_format: 'parquet'
  compression: 'snappy'
  max_checkpoints_per_partition: 10
  cleanup_old_checkpoints: true
```

**Storage Format Comparison:**
- **Parquet**: Best compression (3-5x), fast I/O, production-ready
- **Arrow**: Memory-efficient, zero-copy reads, in-memory processing
- **JSONL**: Human-readable, universal compatibility, debugging

### Event Logging Configuration

```yaml
# Event logging settings
event_logging:
  enabled: true
  log_level: 'INFO'                # DEBUG, INFO, WARNING, ERROR
  max_log_size_mb: 100
  backup_count: 5
  log_to_console: true
  log_to_file: true
```

### Performance Configuration

```yaml
# Performance tuning
performance:
  batch_size: 1000
  prefetch_factor: 2
  num_workers: 4
  memory_limit_gb: 8
  enable_compression: true
  use_arrow_batches: true
  arrow_batch_size: 1000
```

## Usage Examples

### 1. Basic Processing

```python
from data_juicer.config import init_configs
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor

# Load configuration
cfg = init_configs('config.yaml')

# Create executor
executor = PartitionedRayExecutor(cfg)

# Run processing
result_dataset = executor.run()

# Access results
print(f"Processed {len(result_dataset)} samples")
```

### 2. Real-time Monitoring

```python
# Monitor events in real-time
for event in executor.monitor_events():
    print(f"[{event.timestamp:.3f}] {event.event_type.value}: {event.message}")
    
    if event.event_type == EventType.OPERATION_ERROR:
        print(f"Error: {event.error_message}")
```

### 3. Event Analysis

```python
# Get all events
events = executor.get_events()

# Filter by type
partition_events = executor.get_events(event_type=EventType.PARTITION_COMPLETE)
print(f"Completed partitions: {len(partition_events)}")

# Get performance for specific operation
filter_perf = executor.get_performance_summary(operation_name="text_length_filter")
print(f"Filter performance: {filter_perf}")

# Generate comprehensive report
report = executor.generate_status_report()
print(report)
```

### 4. Command Line Usage

```bash
# Basic demo
python demos/partition_and_checkpoint/comprehensive_partitioning_demo.py

# With custom dataset
python demos/partition_and_checkpoint/comprehensive_partitioning_demo.py --dataset data/my_dataset.jsonl

# With custom configuration
python demos/partition_and_checkpoint/comprehensive_partitioning_demo.py --config my_config.yaml

# With analysis
python demos/partition_and_checkpoint/comprehensive_partitioning_demo.py --analyze
```

## Monitoring and Debugging

### 1. Real-time Status Monitoring

```python
# Get current status
status = executor.get_status_summary()
print(f"Success Rate: {status['success_rate']:.1%}")
print(f"Active Partitions: {status['active_partitions']}")
print(f"Completed Partitions: {status['completed_partitions']}")

# Monitor specific partition
partition_status = executor.get_partition_status(partition_id=0)
print(f"Partition Status: {partition_status['status']}")
```

### 2. Event Analysis

```python
# Get all events
events = executor.get_events()

# Filter by event type
partition_events = executor.get_events(event_type=EventType.PARTITION_COMPLETE)
operation_events = executor.get_events(event_type=EventType.OPERATION_START)

# Filter by partition
partition_events = executor.get_events(partition_id=0)

# Filter by time range
recent_events = executor.get_events(start_time=time.time() - 3600)
```

### 3. Checkpoint Analysis

```python
# Get latest checkpoint for partition
checkpoint = executor.checkpoint_manager.get_latest_checkpoint(partition_id=0)

# Load checkpoint data
data = executor.checkpoint_manager.load_checkpoint(checkpoint)

# List all checkpoints
checkpoints = executor.checkpoint_manager.list_checkpoints(partition_id=0)
```

### 4. Performance Analysis

```python
# Get performance summary
perf_summary = executor.get_performance_summary()
print(f"Total Processing Time: {perf_summary['total_time']:.2f}s")
print(f"Average Partition Time: {perf_summary['avg_partition_time']:.2f}s")

# Get operation-specific performance
op_perf = executor.get_performance_summary(operation_name="text_length_filter")
print(f"Filter Operation: {op_perf}")
```

## Best Practices

### 1. Partitioning Strategy

- **Start Small**: Begin with smaller partitions (1,000-10,000 samples) and adjust based on performance
- **Consider Memory**: Ensure partitions fit in available memory (typically 128MB-1GB)
- **Balance Load**: Aim for partitions of similar size for better load balancing
- **Monitor Performance**: Track partition processing times and adjust accordingly

### 2. Checkpointing Strategy

- **Enable for Long Pipelines**: Use checkpoints for pipelines with many operations
- **Choose Storage Format**: Use Parquet for production (compression + performance)
- **Clean Up Regularly**: Enable automatic checkpoint cleanup to save disk space
- **Monitor Disk Usage**: Track checkpoint storage usage

### 3. Fault Tolerance

- **Set Reasonable Retries**: 2-3 retries usually sufficient
- **Monitor Failures**: Track failure patterns to identify systemic issues
- **Use Checkpoints**: Enable checkpoint recovery for better fault tolerance
- **Handle Partial Failures**: Design pipelines to handle partial failures gracefully

### 4. Performance Optimization

- **Use Parquet**: Best balance of compression and performance
- **Enable Compression**: Use Snappy compression for checkpoints
- **Optimize Batch Size**: Adjust batch size based on memory and performance
- **Monitor Resources**: Track CPU, memory, and disk usage

### 5. Monitoring and Debugging

- **Enable Event Logging**: Always enable event logging for production
- **Set Up Alerts**: Monitor for high failure rates or performance issues
- **Regular Analysis**: Periodically analyze event logs for patterns
- **Keep Logs**: Retain logs for debugging and compliance

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Symptoms**: OutOfMemoryError, slow processing, high memory usage
**Solutions**:
- Reduce partition size (`partition_size`)
- Enable checkpoint cleanup (`cleanup_old_checkpoints: true`)
- Use Parquet format for better compression
- Increase available memory or reduce `memory_limit_gb`

#### 2. Disk Space Issues

**Symptoms**: DiskFullError, checkpoint failures, storage warnings
**Solutions**:
- Enable checkpoint cleanup (`cleanup_old_checkpoints: true`)
- Use compression (`compression: 'snappy'`)
- Monitor disk usage in work directory
- Clean up old work directories

#### 3. High Failure Rate

**Symptoms**: Many failed partitions, low success rate, retry loops
**Solutions**:
- Check operation configuration and data quality
- Review error logs in event files
- Increase retry count (`max_retries`)
- Verify dataset format and schema

#### 4. Slow Processing

**Symptoms**: Long processing times, low throughput, resource bottlenecks
**Solutions**:
- Optimize partition size based on available memory
- Use more workers (`num_workers`)
- Enable operator fusion
- Use efficient storage formats (Parquet/Arrow)

#### 5. Event Logging Issues

**Symptoms**: Missing events, log corruption, high log file sizes
**Solutions**:
- Check log rotation settings (`max_log_size_mb`, `backup_count`)
- Verify disk space for log files
- Check log level configuration
- Monitor log file growth

### Debugging Steps

1. **Check Event Logs**: Review processing events for errors
   ```python
   error_events = executor.get_events(event_type=EventType.OPERATION_ERROR)
   for event in error_events:
       print(f"Error in {event.operation_name}: {event.error_message}")
   ```

2. **Analyze Failed Partitions**: Examine failed partition details
   ```python
   failed_partitions = executor.get_events(event_type=EventType.PARTITION_ERROR)
   for event in failed_partitions:
       print(f"Partition {event.partition_id} failed: {event.error_message}")
   ```

3. **Verify Checkpoints**: Check checkpoint availability and integrity
   ```python
   checkpoints = executor.checkpoint_manager.list_checkpoints(partition_id=0)
   print(f"Available checkpoints: {len(checkpoints)}")
   ```

4. **Monitor Resources**: Track CPU, memory, and disk usage
   ```python
   perf_summary = executor.get_performance_summary()
   print(f"Resource usage: {perf_summary['resource_usage']}")
   ```

5. **Review Configuration**: Verify configuration settings
   ```python
   print(f"Current config: {executor.config}")
   ```

### Getting Help

- Check the work directory for detailed logs and reports
- Review event logs for specific error messages
- Analyze checkpoint data for data quality issues
- Monitor system resources for performance bottlenecks
- Use the comprehensive status report for system overview

## Work Directory Structure

The work directory contains all processing artifacts and is organized as follows:

```
work_dir/
├── metadata/
│   ├── dataset_mapping.json      # Partition mapping information
│   └── final_mapping_report.json # Final processing report
├── logs/
│   ├── processing_events.jsonl   # Event log (JSONL format)
│   ├── processing_summary.json   # Processing summary
│   └── performance_metrics.json  # Performance metrics
├── checkpoints/
│   └── partition_000000/
│       ├── op_000_whitespace_normalization_mapper.parquet
│       ├── op_001_text_length_filter.parquet
│       └── metadata.json         # Checkpoint metadata
├── partitions/
│   ├── partition_000000.parquet  # Original partitions
│   └── partition_000001.parquet
├── results/
│   ├── partition_000000_processed.parquet  # Processed partitions
│   └── partition_000001_processed.parquet
└── temp/                         # Temporary files
    ├── ray_objects/
    └── intermediate_data/
```

### Key Files

- **`metadata/dataset_mapping.json`**: Complete partition mapping and metadata
- **`logs/processing_events.jsonl`**: All processing events in JSONL format
- **`logs/processing_summary.json`**: Final processing summary and statistics
- **`checkpoints/`**: Operation-level checkpoints for fault recovery
- **`partitions/`**: Original dataset partitions
- **`results/`**: Final processed partitions

### Log File Analysis

```python
# Analyze event logs
import json

with open('work_dir/logs/processing_events.jsonl', 'r') as f:
    for line in f:
        event = json.loads(line)
        if event['event_type'] == 'OPERATION_ERROR':
            print(f"Error: {event['error_message']}")

# Load processing summary
with open('work_dir/logs/processing_summary.json', 'r') as f:
    summary = json.load(f)
    print(f"Success rate: {summary['success_rate']:.1%}")
```

## Conclusion

The Data-Juicer partitioning and checkpointing system provides a robust, scalable, and observable solution for processing large datasets. By following the best practices outlined in this guide, you can build reliable data processing pipelines that handle failures gracefully and provide detailed insights into processing performance.

For more information, refer to:
- [Partitioning_Checkpointing_EventLogging_Architecture.md](Partitioning_Checkpointing_EventLogging_Architecture.md) - Complete architecture documentation
- [Partitioning_Checkpointing_EventLogging_Summary.md](Partitioning_Checkpointing_EventLogging_Summary.md) - Executive overview
- [Ray_Partitioning_Optimization.md](../demos/partition_and_checkpoint/Ray_Partitioning_Optimization.md) - Ray-specific optimization
- [Universal_Event_Logging_Guide.md](../demos/partition_and_checkpoint/Universal_Event_Logging_Guide.md) - Event logging system 