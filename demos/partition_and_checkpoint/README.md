# DataJuicer Fault-Tolerant Processing with Checkpointing and Event Logging

This directory contains the implementation of fault-tolerant, resumable DataJuicer processing with comprehensive checkpointing, partitioning, and event logging capabilities.

## 🚀 Features Implemented

### ✅ Core Features
- **Job-Specific Directory Isolation**: Each job gets its own dedicated directory structure
- **Flexible Storage Architecture**: Separate storage paths for event logs (fast storage) and checkpoints (large capacity storage)
- **Configurable Checkpointing Strategies**: Multiple checkpointing frequencies and strategies
- **Spark-Style Event Logging**: Comprehensive event tracking in JSONL format for resumability
- **Job Resumption Capabilities**: Resume failed or interrupted jobs from the last checkpoint
- **Comprehensive Job Management**: Job summaries, metadata tracking, and resumption commands

### ✅ Checkpointing Strategies
- `EVERY_OP`: Checkpoint after every operation (most resilient, slower)
- `EVERY_PARTITION`: Checkpoint only at partition completion (balanced)
- `EVERY_N_OPS`: Checkpoint after every N operations (configurable)
- `MANUAL`: Checkpoint only after specified operations
- `DISABLED`: Disable checkpointing entirely

### ✅ Event Logging
- **Human-readable logs**: Loguru-based logging for debugging and monitoring
- **Machine-readable logs**: JSONL format for programmatic analysis and resumption
- **Comprehensive event types**: Job start/complete/failed, partition events, operation events, checkpoint events
- **Real-time monitoring**: Live event streaming and status reporting

### ✅ Job Management
- **Meaningful Job IDs**: Format: `{YYYYMMDD}_{HHMMSS}_{config_name}_{unique_suffix}`
- **Job Summary Files**: Comprehensive metadata for each job run
- **Resumption Commands**: Automatic generation of exact commands to resume jobs
- **Job Validation**: Validation of job resumption parameters and existing state

## 📁 Directory Structure

```
{work_dir}/
├── {job_id}/                    # Job-specific directory
│   ├── job_summary.json         # Job metadata and resumption info
│   ├── metadata/                # Job metadata files
│   │   ├── dataset_mapping.json
│   │   └── final_mapping_report.json
│   ├── partitions/              # Input data partitions
│   ├── intermediate/            # Intermediate processing results
│   └── results/                 # Final processing results
├── {event_log_dir}/{job_id}/    # Flexible event log storage
│   └── event_logs/
│       ├── events.jsonl         # Machine-readable events
│       └── events.log           # Human-readable logs
└── {checkpoint_dir}/{job_id}/   # Flexible checkpoint storage
    ├── checkpoint_*.json        # Checkpoint metadata
    └── partition_*_*.parquet    # Partition checkpoints
```

## 🛠️ Configuration

### Configuration Structure

The configuration uses a **logical nested structure** that groups related settings by concern:

#### New Logical Structure (Recommended)
```yaml
# Partitioning configuration
partition:
  size: 1000  # Number of samples per partition
  max_size_mb: 64  # Maximum partition size in MB

# Fault tolerance configuration
fault_tolerance:
  enabled: true
  max_retries: 3
  retry_backoff: "exponential"  # exponential, linear, fixed

# Intermediate storage configuration for partition and checkpoint data (format, compression, and lifecycle management)
intermediate_storage:
  # File format and compression
  format: "parquet"  # parquet, arrow, jsonl
  compression: "snappy"  # snappy, gzip, none
  use_arrow_batches: true
  arrow_batch_size: 500
  arrow_memory_mapping: false
  
  # File lifecycle management
  preserve_intermediate_data: true  # Keep temporary files for debugging/resumption
  cleanup_temp_files: true
  cleanup_on_success: false
  retention_policy: "keep_all"  # keep_all, keep_failed_only, cleanup_all
  max_retention_days: 7
```

#### Legacy Flat Structure (Still Supported)
```yaml
# Legacy flat configuration (still works)
partition_size: 1000
max_partition_size_mb: 64
enable_fault_tolerance: true
max_retries: 3
preserve_intermediate_data: true
storage_format: "parquet"
use_arrow_batches: true
arrow_batch_size: 500
arrow_memory_mapping: false
```

**Note**: The system reads from the new nested sections first, then falls back to the legacy flat configuration if not found.

### Configuration Sections Explained

#### `partition` - Partitioning and Resilience
Controls how the dataset is split and how failures are handled:
- **Auto-Configuration** (Recommended):
  - `auto_configure`: Enable automatic partition size optimization based on data modality
- **Manual Partitioning** (when `auto_configure: false`):
  - `size`: Number of samples per partition
    - **50-100**: Debugging, quick iterations, small datasets
    - **100-300**: Production, good balance of fault tolerance and efficiency ⭐
    - **300-500**: Large datasets with stable processing
    - **500+**: Only for very large datasets with minimal failure risk
  - `max_size_mb`: Maximum partition size in MB
- **Fault Tolerance**:
  - `enable_fault_tolerance`: Enable/disable retry logic
  - `max_retries`: Maximum retry attempts per partition
  - `retry_backoff`: Retry strategy (`exponential`, `linear`, `fixed`)

#### `intermediate_storage` - Intermediate Data Management
Controls file formats, compression, and lifecycle management for intermediate data:
- **File Format & Compression**:
  - `format`: Storage format (`parquet`, `arrow`, `jsonl`)
  - `compression`: Compression algorithm (`snappy`, `gzip`, `none`)
  - `use_arrow_batches`: Use Arrow batch processing
  - `arrow_batch_size`: Arrow batch size
  - `arrow_memory_mapping`: Enable memory mapping
- **File Lifecycle Management**:
  - `preserve_intermediate_data`: Keep temporary files for debugging
  - `cleanup_temp_files`: Enable automatic cleanup
  - `cleanup_on_success`: Clean up even on successful completion
  - `retention_policy`: File retention strategy (`keep_all`, `keep_failed_only`, `cleanup_all`)
  - `max_retention_days`: Auto-cleanup after X days

### Basic Configuration
```yaml
# Enable fault-tolerant processing
executor_type: ray_partitioned

# Job management
job_id: my_experiment_001  # Optional: auto-generated if not provided

# Checkpointing configuration
checkpoint:
  enabled: true
  strategy: every_op  # every_op, every_partition, every_n_ops, manual, disabled
  n_ops: 2            # For every_n_ops strategy
  op_names:           # For manual strategy
    - clean_links_mapper
    - whitespace_normalization_mapper

# Event logging configuration
event_logging:
  enabled: true
  max_log_size_mb: 100
  backup_count: 5

# Flexible storage paths
event_log_dir: /tmp/fast_event_logs      # Fast storage for event logs
checkpoint_dir: /tmp/large_checkpoints   # Large capacity storage for checkpoints

# Partitioning configuration
partition:
  # Basic partitioning settings
  # Recommended partition sizes:
  # - 50-100: For debugging, quick iterations, small datasets
  # - 100-300: For production, good balance of fault tolerance and efficiency
  # - 300-500: For large datasets with stable processing
  # - 500+: Only for very large datasets with minimal failure risk
  size: 200  # Number of samples per partition (smaller for better fault tolerance)
  max_size_mb: 32  # Maximum partition size in MB (reduced for faster processing)
  
  # Fault tolerance settings
  enable_fault_tolerance: true
  max_retries: 3
  retry_backoff: "exponential"  # exponential, linear, fixed

# Intermediate storage configuration for partition and checkpoint data (format, compression, and lifecycle management)
intermediate_storage:
  # File format and compression
  format: "parquet"  # parquet, arrow, jsonl
  compression: "snappy"  # snappy, gzip, none
  use_arrow_batches: true
  arrow_batch_size: 500
  arrow_memory_mapping: false
  
  # File lifecycle management
  preserve_intermediate_data: true  # Keep temporary files for debugging/resumption
  cleanup_temp_files: true
  cleanup_on_success: false
  retention_policy: "keep_all"  # keep_all, keep_failed_only, cleanup_all
  max_retention_days: 7
```

## 🚀 Quick Start

### 1. Basic Usage
```bash
# Run with auto-generated job ID
dj-process --config configs/demo/checkpoint_config_example.yaml

# Run with custom job ID
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id my_experiment_001
```

### 2. Resume a Job
```bash
# Resume using the job ID
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id my_experiment_001
```

### 3. Different Checkpoint Strategies
```bash
# Checkpoint every partition
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id partition_test --checkpoint.strategy every_partition

# Checkpoint every 3 operations
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id n_ops_test --checkpoint.strategy every_n_ops --checkpoint.n_ops 3

# Manual checkpointing
dj-process --config configs/demo/checkpoint_config_example.yaml --job_id manual_test --checkpoint.strategy manual --checkpoint.op_names clean_links_mapper,whitespace_normalization_mapper
```

### 4. Run Comprehensive Demo
```bash
# Run the full demo showcasing all features
python demos/partition_and_checkpoint/run_comprehensive_demo.py
```

## 📊 Monitoring and Debugging

### View Job Information
```bash
# Check job summary
cat ./outputs/demo-checkpoint-strategies/{job_id}/job_summary.json

# View event logs
cat /tmp/fast_event_logs/{job_id}/event_logs/events.jsonl

# View human-readable logs
cat /tmp/fast_event_logs/{job_id}/event_logs/events.log
```

### List Available Jobs
```bash
# List all job directories
ls -la ./outputs/demo-checkpoint-strategies/
```

### Check Flexible Storage
```bash
# Check event logs in fast storage
ls -la /tmp/fast_event_logs/

# Check checkpoints in large storage
ls -la /tmp/large_checkpoints/
```

## 🤖 Auto-Configuration System

### **Smart Partition Sizing by Modality**

DataJuicer now includes an intelligent auto-configuration system that automatically determines optimal partition sizes based on your data characteristics:

#### **How It Works**

1. **Modality Detection**: Analyzes your dataset to detect the primary modality (text, image, audio, video, multimodal)
2. **Dataset Analysis**: Examines sample characteristics (text length, media counts, file sizes)
3. **Pipeline Complexity**: Considers the complexity of your processing operations
4. **Resource Optimization**: Adjusts partition sizes for optimal memory usage and fault tolerance

#### **Modality-Specific Optimizations**

| Modality | Default Size | Max Size | Memory Multiplier | Use Case |
|----------|--------------|----------|-------------------|----------|
| **Text** | 200 samples | 1000 | 1.0x | Efficient processing, low memory |
| **Image** | 50 samples | 200 | 5.0x | Moderate memory, image processing |
| **Audio** | 30 samples | 100 | 8.0x | High memory, audio processing |
| **Video** | 10 samples | 50 | 20.0x | Very high memory, complex processing |
| **Multimodal** | 20 samples | 100 | 10.0x | Multiple modalities, moderate complexity |

#### **Enable Auto-Configuration**

```yaml
partition:
  auto_configure: true  # Enable automatic optimization
  # Manual settings are ignored when auto_configure is true
  size: 200
  max_size_mb: 32
```

#### **Manual Override**

```yaml
partition:
  auto_configure: false  # Disable auto-configuration
  size: 100  # Use your own partition size
  max_size_mb: 64
```

## 📊 Partition Sizing Guidelines

### **Why Smaller Partitions Are Better**

**Fault Tolerance**: Smaller partitions mean smaller units of failure. If a partition fails, you lose less work.

**Recovery Speed**: Failed partitions can be retried faster, reducing overall job time.

**Progress Visibility**: More granular progress tracking and faster feedback.

**Memory Efficiency**: Lower memory usage per partition, better for resource-constrained environments.

**Debugging**: Easier to isolate and debug issues in smaller chunks.

### **Partition Size Recommendations**

| Use Case | Partition Size | When to Use |
|----------|---------------|-------------|
| **Debugging** | 50-100 samples | Quick iterations, testing, small datasets |
| **Production** ⭐ | 100-300 samples | Most use cases, good balance |
| **Large Datasets** | 300-500 samples | Stable processing, large datasets |
| **Very Large** | 500+ samples | Only when failure risk is minimal |

### **Factors to Consider**

- **Dataset Size**: Larger datasets can use larger partitions
- **Processing Complexity**: Complex operations benefit from smaller partitions
- **Failure Rate**: Higher failure rates need smaller partitions
- **Memory Constraints**: Limited memory requires smaller partitions
- **Time Sensitivity**: Faster feedback needs smaller partitions

## 🔧 Implementation Details

### Core Components

1. **`EventLoggingMixin`** (`data_juicer/core/executor/event_logging_mixin.py`)
   - Provides event logging capabilities to executors
   - Manages job-specific directories and flexible storage
   - Handles job summary creation and validation
   - Implements Spark-style event logging schema

2. **`PartitionedRayExecutor`** (`data_juicer/core/executor/ray_executor_partitioned.py`)
   - Extends Ray executor with partitioning and fault tolerance
   - Implements configurable checkpointing strategies
   - Integrates with EventLoggingMixin for comprehensive logging
   - Handles job resumption from checkpoints

3. **Configuration Integration** (`data_juicer/config/config.py`)
   - Added command-line arguments for job management
   - Added checkpointing configuration options
   - Added flexible storage path configuration

### Event Types
- `JOB_START`, `JOB_COMPLETE`, `JOB_FAILED`
- `PARTITION_START`, `PARTITION_COMPLETE`, `PARTITION_FAILED`
- `OP_START`, `OP_COMPLETE`, `OP_FAILED`
- `CHECKPOINT_SAVE`, `CHECKPOINT_LOAD`
- `PROCESSING_START`, `PROCESSING_COMPLETE`, `PROCESSING_ERROR`
- `RESOURCE_USAGE`, `PERFORMANCE_METRIC`
- `WARNING`, `INFO`, `DEBUG`

## 🎯 Use Cases

### 1. Large Dataset Processing
- Process datasets that are too large for memory
- Automatic partitioning with fault tolerance
- Resume processing after failures

### 2. Experimental Workflows
- Track different experiments with meaningful job IDs
- Compare results across different configurations
- Maintain experiment history and reproducibility

### 3. Production Pipelines
- Robust error handling and recovery
- Comprehensive monitoring and logging
- Flexible storage for different performance requirements

### 4. Research and Development
- Iterative development with checkpoint resumption
- Detailed event logging for analysis
- Configurable checkpointing for different scenarios

## 🔍 Troubleshooting

### Common Issues

1. **Job resumption fails**
   - Check if job summary exists: `ls -la ./outputs/{work_dir}/{job_id}/job_summary.json`
   - Verify checkpoint files exist: `ls -la /tmp/large_checkpoints/{job_id}/`

2. **Event logs not found**
   - Check flexible storage paths: `ls -la /tmp/fast_event_logs/{job_id}/`
   - Verify event logging is enabled in config

3. **Checkpointing not working**
   - Verify checkpoint strategy in config
   - Check if checkpoint directory is writable
   - Ensure checkpoint.enabled is true

4. **Performance issues**
   - Adjust partition size based on available memory
   - Consider different checkpoint strategies
   - Use appropriate storage formats (parquet for large datasets)

### Debug Commands
```bash
# Check Ray cluster status
ray status

# View Ray dashboard
open http://localhost:8265

# Check DataJuicer logs
tail -f /tmp/fast_event_logs/{job_id}/event_logs/events.log
```

## 📊 Understanding Intermediate Data

### What is Intermediate Data?

Intermediate data refers to temporary results generated during the processing pipeline that exist between operations and before the final output. In DataJuicer's partitioned processing, this includes:

1. **Partition-level intermediate data**: Results after each operation within a partition
2. **Operation-level intermediate data**: Data that exists between operations (e.g., after `clean_links_mapper` but before `whitespace_normalization_mapper`)
3. **Checkpoint intermediate data**: Temporary files created during checkpointing

### When to Preserve Intermediate Data

**Enable `preserve_intermediate_data: true` when you need:**
- **Debugging**: Inspect what the data looks like after each operation
- **Resumption**: If a job fails, see exactly where it failed and what the data looked like
- **Analysis**: Understand how each operation transforms the data
- **Development**: Iterate on processing pipelines with detailed inspection

**Disable `preserve_intermediate_data: false` when you want:**
- **Performance**: Faster processing with less disk I/O
- **Storage efficiency**: Reduced disk space usage
- **Production**: Clean processing without temporary file accumulation

### Example Directory Structure with Intermediate Data

```
{job_dir}/intermediate/
├── partition_000000/
│   ├── op_000_clean_links_mapper.parquet      # After clean_links_mapper
│   ├── op_001_clean_email_mapper.parquet      # After clean_email_mapper
│   ├── op_002_whitespace_normalization_mapper.parquet
│   └── op_003_fix_unicode_mapper.parquet      # After fix_unicode_mapper
└── partition_000001/
    ├── op_000_clean_links_mapper.parquet
    └── ...
```

## 📈 Performance Considerations

### Checkpointing Overhead
- `EVERY_OP`: Highest overhead, maximum resilience
- `EVERY_PARTITION`: Balanced overhead and resilience
- `EVERY_N_OPS`: Configurable overhead
- `MANUAL`: Minimal overhead, requires careful planning

### Storage Recommendations
- **Event logs**: Use fast storage (SSD) for real-time monitoring
- **Checkpoints**: Use large capacity storage (HDD/network storage) for cost efficiency
- **Partitions**: Use local storage for processing speed

### Memory Management
- Adjust `partition_size` based on available memory
- Use `max_partition_size_mb` to limit partition size
- Consider `preserve_intermediate_data` for debugging vs. performance

## 🎉 Success Metrics

The implementation successfully demonstrates:
- ✅ **Fault Tolerance**: Jobs can resume after failures
- ✅ **Scalability**: Handles large datasets through partitioning
- ✅ **Observability**: Comprehensive logging and monitoring
- ✅ **Flexibility**: Configurable checkpointing and storage
- ✅ **Usability**: Simple command-line interface with meaningful job IDs
- ✅ **Performance**: Fast resumption from checkpoints
- ✅ **Reliability**: Robust error handling and validation

## 🔮 Future Enhancements

Potential areas for future development:
- **Distributed checkpointing**: Multi-node checkpoint coordination
- **Incremental checkpointing**: Only save changed data
- **Checkpoint compression**: Reduce storage requirements
- **Advanced monitoring**: Web-based dashboard for job monitoring
- **Checkpoint versioning**: Support for multiple checkpoint versions
- **Integration with external systems**: Cloud storage, monitoring systems 