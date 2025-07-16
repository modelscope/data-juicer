# Universal Event Logging for Data-Juicer Executors

## Overview

The Event Logging system provides comprehensive monitoring, debugging, and observability capabilities that can be used with **any** Data-Juicer executor (default, ray, ray_partitioned, etc.). This universal approach ensures consistent monitoring and debugging capabilities across all execution modes.

## Why Universal Event Logging Makes Sense

### 1. **Consistent Observability**
- **Same Interface**: All executors provide the same event logging interface
- **Unified Monitoring**: Consistent monitoring regardless of execution mode
- **Standardized Debugging**: Same debugging tools work across all executors
- **Performance Analysis**: Comparable performance metrics across execution modes

### 2. **Production Benefits**
- **Alerting**: Set up alerts based on event patterns for any executor
- **Troubleshooting**: Quick identification of issues in production
- **Capacity Planning**: Understand resource usage patterns
- **Quality Assurance**: Track data quality metrics consistently

### 3. **Development Benefits**
- **Debugging**: Detailed operation tracking for development
- **Performance Optimization**: Identify bottlenecks in any processing pipeline
- **Testing**: Comprehensive event logs for testing and validation
- **Documentation**: Automatic documentation of processing steps

## Architecture

### Event Logging Mixin
The `EventLoggingMixin` provides event logging capabilities that can be easily added to any executor:

```python
class EventLoggingDefaultExecutor(EventLoggingMixin, DefaultExecutor):
    """Default executor with event logging capabilities."""
    pass

class EventLoggingRayExecutor(EventLoggingMixin, RayExecutor):
    """Ray executor with event logging capabilities."""
    pass

class EventLoggingPartitionedRayExecutor(EventLoggingMixin, PartitionedRayExecutor):
    """Partitioned Ray executor with event logging capabilities."""
    pass
```

### Event Types
The system tracks various types of events:

- **Operation Events**: Start, completion, and errors of operations
- **Partition Events**: Partition processing status (for partitioned executors)
- **Dataset Events**: Loading and saving of datasets
- **Processing Events**: Overall pipeline processing status
- **Resource Events**: CPU, memory, and I/O usage
- **Performance Events**: Throughput and timing metrics
- **System Events**: Warnings, info, and debug messages

## Configuration

### Basic Event Logging Configuration
```yaml
# Enable event logging for any executor
event_logging:
  enabled: true                    # Enable/disable event logging
  log_level: "INFO"               # Log level: DEBUG, INFO, WARNING, ERROR
  max_log_size_mb: 100            # Maximum log file size before rotation
  backup_count: 5                 # Number of backup log files to keep
```

### Complete Configuration Example
```yaml
# Basic executor configuration
executor_type: "default"  # Can be: default, ray, ray_partitioned
work_dir: "./work_dir"

# Event logging configuration
event_logging:
  enabled: true
  log_level: "INFO"
  max_log_size_mb: 100
  backup_count: 5

# Processing pipeline
process:
  - name: "text_length_filter"
    args:
      min_len: 10
      max_len: 1000
```

## Usage Examples

### 1. Default Executor with Event Logging
```python
from data_juicer.config import init_configs
from data_juicer.core.executor.default_executor import DefaultExecutor
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType

class EventLoggingDefaultExecutor(EventLoggingMixin, DefaultExecutor):
    pass

# Load configuration with event logging enabled
cfg = init_configs()
cfg.event_logging = {'enabled': True}

# Create executor with event logging
executor = EventLoggingDefaultExecutor(cfg)

# Run processing
result = executor.run()

# Get events
events = executor.get_events()
print(f"Logged {len(events)} events")

# Get performance summary
perf_summary = executor.get_performance_summary()
print(f"Average duration: {perf_summary.get('avg_duration', 0):.3f}s")
```

### 2. Ray Executor with Event Logging
```python
from data_juicer.core.executor.ray_executor import RayExecutor
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin

class EventLoggingRayExecutor(EventLoggingMixin, RayExecutor):
    pass

# Create Ray executor with event logging
executor = EventLoggingRayExecutor(cfg)

# Monitor events in real-time
for event in executor.monitor_events():
    print(f"[{event.timestamp:.3f}] {event.event_type.value}: {event.message}")
```

### 3. Partitioned Ray Executor with Event Logging
```python
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin

class EventLoggingPartitionedRayExecutor(EventLoggingMixin, PartitionedRayExecutor):
    pass

# Create partitioned Ray executor with event logging
executor = EventLoggingPartitionedRayExecutor(cfg)

# Get partition-specific events
partition_events = executor.get_events(event_type=EventType.PARTITION_COMPLETE)
print(f"Completed partitions: {len(partition_events)}")
```

## Event Analysis and Monitoring

### 1. Real-time Event Monitoring
```python
# Monitor all events in real-time
for event in executor.monitor_events():
    print(f"[{event.timestamp:.3f}] {event.event_type.value}: {event.message}")

# Monitor specific event types
for event in executor.monitor_events(event_type=EventType.OPERATION_ERROR):
    print(f"Error: {event.error_message}")
```

### 2. Event Filtering and Querying
```python
# Get all events
all_events = executor.get_events()

# Get events by type
operation_events = executor.get_events(event_type=EventType.OPERATION_START)
error_events = executor.get_events(event_type=EventType.OPERATION_ERROR)

# Get events by time range
recent_events = executor.get_events(start_time=time.time() - 3600)  # Last hour

# Get events by operation
filter_events = executor.get_events(operation_name="text_length_filter")

# Get recent events with limit
recent_events = executor.get_events(limit=10)
```

### 3. Performance Analysis
```python
# Get overall performance summary
perf_summary = executor.get_performance_summary()
print(f"Total operations: {perf_summary.get('total_operations', 0)}")
print(f"Average duration: {perf_summary.get('avg_duration', 0):.3f}s")
print(f"Average throughput: {perf_summary.get('avg_throughput', 0):.1f} samples/s")

# Get performance for specific operation
filter_perf = executor.get_performance_summary(operation_name="text_length_filter")
print(f"Filter performance: {filter_perf}")
```

### 4. Status Reporting
```python
# Generate comprehensive status report
report = executor.generate_status_report()
print(report)

# Example output:
# === EVENT LOGGING STATUS REPORT ===
# Total Events: 25
# Errors: 0
# Warnings: 2
# 
# Event Type Distribution:
#   operation_start: 8 (32.0%)
#   operation_complete: 8 (32.0%)
#   processing_start: 1 (4.0%)
#   processing_complete: 1 (4.0%)
#   ...
# 
# Performance Summary:
#   Total Operations: 8
#   Average Duration: 0.125s
#   Average Throughput: 800.0 samples/s
```

## Event Log Files

### File Structure
```
work_dir/
├── event_logs/
│   ├── events.log              # Current log file
│   ├── events.log.1.gz         # Compressed backup
│   ├── events.log.2.gz         # Compressed backup
│   └── ...
```

### Log Format
```
2024-01-15 10:30:45.123 | INFO | EVENT[operation_start] | TIME[2024-01-15T10:30:45.123] | OP[text_length_filter] | MSG[Starting operation: text_length_filter]
2024-01-15 10:30:45.456 | INFO | EVENT[operation_complete] | TIME[2024-01-15T10:30:45.456] | OP[text_length_filter] | DURATION[0.333] | MSG[Completed operation: text_length_filter in 0.333s]
```

## Integration with Existing Executors

### 1. Default Executor Integration
The default executor can be enhanced with event logging by simply adding the mixin:

```python
# Before: Basic default executor
executor = DefaultExecutor(cfg)

# After: Default executor with event logging
executor = EventLoggingDefaultExecutor(cfg)
```

### 2. Ray Executor Integration
Ray executors can be enhanced similarly:

```python
# Before: Basic Ray executor
executor = RayExecutor(cfg)

# After: Ray executor with event logging
executor = EventLoggingRayExecutor(cfg)
```

### 3. Partitioned Ray Executor Integration
The partitioned Ray executor already includes event logging, but it can be enhanced further:

```python
# Before: Basic partitioned Ray executor
executor = PartitionedRayExecutor(cfg)

# After: Enhanced partitioned Ray executor with additional logging
executor = EventLoggingPartitionedRayExecutor(cfg)
```

## Benefits Across Different Execution Modes

### Default Executor Benefits
- **Operation Tracking**: Track each operation's start, completion, and performance
- **Error Debugging**: Detailed error tracking with stack traces
- **Performance Analysis**: Identify slow operations and bottlenecks
- **Resource Monitoring**: Track CPU and memory usage

### Ray Executor Benefits
- **Distributed Monitoring**: Track operations across Ray cluster
- **Resource Utilization**: Monitor cluster resource usage
- **Fault Detection**: Identify node failures and errors
- **Performance Optimization**: Optimize distributed processing

### Partitioned Ray Executor Benefits
- **Partition Tracking**: Monitor individual partition processing
- **Fault Tolerance**: Track partition failures and retries
- **Progress Monitoring**: Real-time progress tracking
- **Checkpoint Analysis**: Monitor checkpoint operations

## Production Deployment

### 1. Configuration for Production
```yaml
event_logging:
  enabled: true
  log_level: "INFO"              # Use INFO for production
  max_log_size_mb: 500           # Larger logs for production
  backup_count: 10               # More backups for production
```

### 2. Monitoring and Alerting
```python
# Set up alerts for error events
error_events = executor.get_events(event_type=EventType.OPERATION_ERROR)
if len(error_events) > 0:
    send_alert(f"Found {len(error_events)} errors in processing")

# Monitor performance degradation
perf_summary = executor.get_performance_summary()
if perf_summary.get('avg_duration', 0) > 10.0:  # More than 10 seconds
    send_alert("Performance degradation detected")
```

### 3. Log Management
- **Log Rotation**: Automatic log rotation prevents disk space issues
- **Compression**: Log files are automatically compressed
- **Retention**: Configurable retention policies
- **Cleanup**: Automatic cleanup of old log files

## Best Practices

### 1. Configuration
- **Enable for Production**: Always enable event logging in production
- **Appropriate Log Level**: Use INFO for production, DEBUG for development
- **Log Size Management**: Set appropriate log size limits
- **Backup Strategy**: Configure sufficient backup files

### 2. Monitoring
- **Real-time Monitoring**: Use real-time event monitoring for immediate feedback
- **Performance Tracking**: Monitor performance metrics regularly
- **Error Analysis**: Analyze error patterns and trends
- **Resource Monitoring**: Track resource usage patterns

### 3. Analysis
- **Regular Reports**: Generate status reports regularly
- **Performance Optimization**: Use performance data to optimize operations
- **Error Prevention**: Use error patterns to prevent future issues
- **Capacity Planning**: Use resource data for capacity planning

## Conclusion

Universal event logging provides consistent monitoring, debugging, and observability capabilities across all Data-Juicer executors. This approach ensures that:

1. **All executors** benefit from comprehensive event tracking
2. **Developers** have consistent debugging tools regardless of execution mode
3. **Production systems** have reliable monitoring and alerting
4. **Performance optimization** is possible across all execution modes
5. **Error handling** is consistent and comprehensive

The event logging system makes Data-Juicer more robust, debuggable, and production-ready across all execution modes. 