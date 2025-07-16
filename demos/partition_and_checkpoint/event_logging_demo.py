#!/usr/bin/env python3
"""
Event Logging Demo for Data-Juicer Executors

This demo shows how the EventLoggingMixin can be used with any executor
(default, ray, etc.) to provide comprehensive event logging, monitoring,
and debugging capabilities.

Features demonstrated:
1. Event logging with default executor
2. Real-time event monitoring
3. Performance metrics tracking
4. Error tracking and debugging
5. Status reporting
6. Event filtering and analysis
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data_juicer.config import init_configs
from data_juicer.core.executor.default_executor import DefaultExecutor
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType


class EventLoggingDefaultExecutor(EventLoggingMixin, DefaultExecutor):
    """Default executor with event logging capabilities."""
    
    def __init__(self, cfg=None):
        """Initialize with event logging."""
        super().__init__(cfg)
    
    def run(self, dataset=None, load_data_np=None, skip_return=False):
        """Run with event logging."""
        try:
            # Log processing start
            self._log_event(EventType.PROCESSING_START, "Starting data processing pipeline")
            
            # Log dataset loading
            if dataset is not None:
                self._log_event(EventType.DATASET_LOAD, f"Using existing dataset with {len(dataset)} samples")
            else:
                self._log_event(EventType.DATASET_LOAD, "Loading dataset from dataset builder")
            
            # Call parent run method
            start_time = time.time()
            result = super().run(dataset, load_data_np, skip_return)
            duration = time.time() - start_time
            
            # Log processing completion
            self._log_event(
                EventType.PROCESSING_COMPLETE, 
                f"Completed processing pipeline in {duration:.3f}s",
                duration=duration,
                metadata={'result_samples': len(result) if result else 0}
            )
            
            return result
            
        except Exception as e:
            # Log processing error
            self._log_operation_error("pipeline", e)
            raise


def create_demo_config():
    """Create a demo configuration with event logging enabled."""
    cfg = init_configs()
    
    # Set up work directory
    work_dir = tempfile.mkdtemp(prefix="event_logging_demo_")
    cfg.work_dir = work_dir
    
    # Enable event logging
    cfg.event_logging = {
        'enabled': True,
        'log_level': 'INFO',
        'max_log_size_mb': 50,
        'backup_count': 3
    }
    
    # Set up a simple dataset configuration
    cfg.dataset_path = "demos/data/demo-dataset_1725870268.jsonl"
    cfg.export_path = os.path.join(work_dir, "output.jsonl")
    
    # Simple processing pipeline
    cfg.process = [
        {
            "name": "text_length_filter",
            "args": {
                "min_len": 10,
                "max_len": 1000
            }
        },
        {
            "name": "text_cleaning",
            "args": {
                "text_key": "text"
            }
        }
    ]
    
    return cfg, work_dir


def demonstrate_event_logging():
    """Demonstrate event logging capabilities."""
    print("=== Event Logging Demo for Data-Juicer Executors ===\n")
    
    # Create configuration
    cfg, work_dir = create_demo_config()
    print(f"Work directory: {work_dir}")
    
    # Create executor with event logging
    executor = EventLoggingDefaultExecutor(cfg)
    
    print("\n1. Running processing pipeline with event logging...")
    
    # Start real-time event monitoring in background
    import threading
    
    def monitor_events():
        print("\n--- Real-time Event Monitor ---")
        for event in executor.monitor_events():
            print(f"[{event.timestamp:.3f}] {event.event_type.value}: {event.message}")
            if event.duration:
                print(f"  Duration: {event.duration:.3f}s")
            if event.error_message:
                print(f"  Error: {event.error_message}")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_events, daemon=True)
    monitor_thread.start()
    
    # Run processing
    try:
        start_time = time.time()
        result = executor.run()
        total_time = time.time() - start_time
        
        print(f"\nProcessing completed in {total_time:.3f}s")
        print(f"Result dataset size: {len(result) if result else 0} samples")
        
    except Exception as e:
        print(f"\nProcessing failed: {e}")
    
    # Wait a bit for events to be logged
    time.sleep(2)
    
    print("\n2. Event Analysis and Reporting...")
    
    # Get all events
    all_events = executor.get_events()
    print(f"Total events logged: {len(all_events)}")
    
    # Get events by type
    operation_events = executor.get_events(event_type=EventType.OPERATION_START)
    print(f"Operation start events: {len(operation_events)}")
    
    error_events = executor.get_events(event_type=EventType.OPERATION_ERROR)
    print(f"Error events: {len(error_events)}")
    
    # Get performance summary
    perf_summary = executor.get_performance_summary()
    if perf_summary:
        print(f"\nPerformance Summary:")
        print(f"  Total operations: {perf_summary.get('total_operations', 0)}")
        print(f"  Average duration: {perf_summary.get('avg_duration', 0):.3f}s")
        print(f"  Average throughput: {perf_summary.get('avg_throughput', 0):.1f} samples/s")
    
    # Generate comprehensive status report
    print(f"\n3. Comprehensive Status Report:")
    print(executor.generate_status_report())
    
    # Show event log file
    event_log_file = os.path.join(work_dir, "event_logs", "events.log")
    if os.path.exists(event_log_file):
        print(f"\n4. Event Log File Location: {event_log_file}")
        print("Recent log entries:")
        with open(event_log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:  # Show last 10 lines
                print(f"  {line.strip()}")
    
    print(f"\n5. Work Directory Contents:")
    work_path = Path(work_dir)
    for item in work_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(work_path)
            print(f"  {rel_path}")
    
    print(f"\n=== Demo Complete ===")
    print(f"Work directory preserved at: {work_dir}")
    print("You can examine the event logs and other files for detailed analysis.")


def demonstrate_event_filtering():
    """Demonstrate event filtering capabilities."""
    print("\n=== Event Filtering Demo ===\n")
    
    # Create configuration
    cfg, work_dir = create_demo_config()
    
    # Create executor
    executor = EventLoggingDefaultExecutor(cfg)
    
    # Run processing to generate events
    try:
        executor.run()
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Demonstrate different filtering options
    print("Event filtering examples:")
    
    # Get recent events
    recent_events = executor.get_events(limit=5)
    print(f"\n1. Recent 5 events: {len(recent_events)}")
    for event in recent_events:
        print(f"  {event.event_type.value}: {event.message}")
    
    # Get events by time range
    current_time = time.time()
    recent_events = executor.get_events(start_time=current_time - 60)  # Last minute
    print(f"\n2. Events in last minute: {len(recent_events)}")
    
    # Get events by operation
    dataset_events = executor.get_events(operation_name="dataset_load")
    print(f"\n3. Dataset load events: {len(dataset_events)}")
    
    # Get error events
    error_events = executor.get_events(event_type=EventType.OPERATION_ERROR)
    print(f"\n4. Error events: {len(error_events)}")
    for event in error_events:
        print(f"  Error: {event.error_message}")
        if event.stack_trace:
            print(f"  Stack trace: {event.stack_trace[:200]}...")


if __name__ == "__main__":
    # Run the main demo
    demonstrate_event_logging()
    
    # Run filtering demo
    demonstrate_event_filtering() 