#!/usr/bin/env python3
"""
Execution Graph Generator for DataJuicer Partitioned Processing

This script analyzes events.jsonl and dag_execution_plan.json to create
a comprehensive execution graph showing the parallel processing flow.
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def find_files_in_work_dir(work_dir):
    """Find the events.jsonl and dag_execution_plan.json files in the work directory."""
    events_file = None
    dag_file = None

    # Look for events.jsonl file
    events_pattern = os.path.join(work_dir, "**", "events.jsonl")
    events_files = glob.glob(events_pattern, recursive=True)
    if events_files:
        events_file = events_files[0]
        print(f"Found events file: {events_file}")
    else:
        print(f"Warning: No events.jsonl file found in {work_dir}")

    # Look for dag_execution_plan.json file
    dag_pattern = os.path.join(work_dir, "**", "dag_execution_plan.json")
    dag_files = glob.glob(dag_pattern, recursive=True)
    if dag_files:
        dag_file = dag_files[0]
        print(f"Found DAG file: {dag_file}")
    else:
        print(f"Warning: No dag_execution_plan.json file found in {work_dir}")

    return events_file, dag_file


def parse_events(events_file):
    """Parse events from JSONL file and extract timing information."""
    events = []
    with open(events_file, "r") as f:
        for line in f:
            events.append(json.loads(line.strip()))
    return events


def parse_dag_plan(dag_file):
    """Parse DAG execution plan from JSON file."""
    with open(dag_file, "r") as f:
        return json.load(f)


def extract_timing_data(events):
    """Extract timing data for operations and partitions."""
    timing_data = {
        "partitions": {},
        "operations": {},
        "checkpoints": {},
        "job_start": None,
        "job_end": None,
        "partition_creation": {},
        "merging": {},
    }

    for event in events:
        event_type = event.get("event_type")
        timestamp = event.get("timestamp")

        if event_type == "job_start":
            timing_data["job_start"] = timestamp
        elif event_type == "job_complete":
            timing_data["job_end"] = timestamp
        elif event_type == "repartition_start":
            timing_data["partition_creation"]["start"] = timestamp
        elif event_type == "repartition_complete":
            timing_data["partition_creation"]["end"] = timestamp
            timing_data["partition_creation"]["metadata"] = event.get("metadata", {})
        elif event_type == "partition_creation_start":
            partition_id = event.get("partition_id")
            if partition_id not in timing_data["partitions"]:
                timing_data["partitions"][partition_id] = {"start": timestamp}
            timing_data["partitions"][partition_id]["creation_start"] = timestamp
        elif event_type == "partition_creation_complete":
            partition_id = event.get("partition_id")
            if partition_id in timing_data["partitions"]:
                timing_data["partitions"][partition_id]["creation_end"] = timestamp
                timing_data["partitions"][partition_id]["creation_metadata"] = event.get("metadata", {})
        elif event_type == "partition_start":
            partition_id = event.get("partition_id")
            if partition_id not in timing_data["partitions"]:
                timing_data["partitions"][partition_id] = {"start": timestamp}
            else:
                timing_data["partitions"][partition_id]["start"] = timestamp
        elif event_type == "partition_complete":
            partition_id = event.get("partition_id")
            if partition_id in timing_data["partitions"]:
                timing_data["partitions"][partition_id]["end"] = timestamp
        elif event_type == "op_start":
            partition_id = event.get("partition_id")
            op_idx = event.get("operation_idx")
            op_name = event.get("operation_name")
            key = f"p{partition_id}_op{op_idx}_{op_name}"
            timing_data["operations"][key] = {
                "partition_id": partition_id,
                "op_idx": op_idx,
                "op_name": op_name,
                "start": timestamp,
            }
        elif event_type == "op_complete":
            partition_id = event.get("partition_id")
            op_idx = event.get("operation_idx")
            op_name = event.get("operation_name")
            key = f"p{partition_id}_op{op_idx}_{op_name}"
            if key in timing_data["operations"]:
                timing_data["operations"][key]["end"] = timestamp
                # Calculate actual duration from start/end timestamps instead of using reported duration
                if "start" in timing_data["operations"][key]:
                    actual_duration = timestamp - timing_data["operations"][key]["start"]
                    timing_data["operations"][key]["duration"] = actual_duration
                else:
                    timing_data["operations"][key]["duration"] = event.get("duration", 0)
                timing_data["operations"][key]["input_rows"] = event.get("input_rows", 0)
                timing_data["operations"][key]["output_rows"] = event.get("output_rows", 0)
        elif event_type == "checkpoint_save":
            partition_id = event.get("partition_id")
            op_idx = event.get("operation_idx")
            op_name = event.get("operation_name")
            key = f"p{partition_id}_op{op_idx}_{op_name}"
            timing_data["checkpoints"][key] = timestamp

    return timing_data


def create_execution_graph(timing_data, dag_plan):
    """Create a comprehensive execution graph visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(24, 18))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 18)
    ax.axis("off")

    # Colors for different operation types
    colors = {
        "clean_links_mapper": "#E3F2FD",  # Light blue
        "clean_email_mapper": "#BBDEFB",  # Medium blue
        "whitespace_normalization_mapper": "#90CAF9",  # Darker blue
        "fix_unicode_mapper": "#64B5F6",  # Even darker blue
        "text_length_filter": "#F3E5F5",  # Light purple
        "alphanumeric_filter": "#E1BEE7",  # Medium purple
        "character_repetition_filter": "#CE93D8",  # Darker purple
        "word_repetition_filter": "#BA68C8",  # Even darker purple
        "checkpoint": "#FFF3E0",  # Light orange
        "partition": "#E8F5E8",  # Light green
        "partition_creation": "#C8E6C9",  # Medium green
        "merging": "#FFCCBC",  # Light red
        "text": "#212121",  # Dark gray
    }

    # Title
    ax.text(12, 17.5, "DataJuicer Partitioned Processing Execution Graph", fontsize=18, fontweight="bold", ha="center")

    # Subtitle with job info
    if timing_data["job_start"]:
        start_time = datetime.fromtimestamp(timing_data["job_start"])
        ax.text(
            12,
            17.1,
            f'Job started at {start_time.strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=14,
            ha="center",
            color="gray",
        )

    # Normalize timestamps to relative time
    if timing_data["job_start"]:
        base_time = timing_data["job_start"]
    else:
        base_time = min([op["start"] for op in timing_data["operations"].values()])

    # Calculate time range
    all_times = []
    for op in timing_data["operations"].values():
        if "start" in op and "end" in op:
            all_times.extend([op["start"], op["end"]])

    if all_times:
        time_range = max(all_times) - min(all_times)
        time_scale = 20 / time_range  # Scale to fit in 20 units width
    else:
        time_scale = 1

    # Draw partition creation phase - aligned with main flow and starting from 0s
    if "partition_creation" in timing_data and "start" in timing_data["partition_creation"]:
        creation_start = timing_data["partition_creation"]["start"]
        creation_end = timing_data["partition_creation"].get("end", creation_start + 15)

        # Align with the main processing flow starting position (x=3.0)
        start_x = 3.0
        end_x = 3.0 + (creation_end - creation_start) * time_scale
        width = end_x - start_x

        creation_box = FancyBboxPatch(
            (start_x, 15.0),
            width,
            0.6,
            boxstyle="round,pad=0.1",
            facecolor=colors["partition_creation"],
            edgecolor="black",
            linewidth=2,
            zorder=15,
        )
        ax.add_patch(creation_box)
        ax.text(
            start_x + width / 2, 15.3, "Dataset Partitioning", fontsize=12, fontweight="bold", ha="center", zorder=16
        )

        if "metadata" in timing_data["partition_creation"]:
            metadata = timing_data["partition_creation"]["metadata"]
            partition_count = metadata.get("partition_count", 0)
            total_samples = metadata.get("total_samples", 0)
            duration = metadata.get("duration_seconds", 0)
            ax.text(
                start_x + width / 2,
                15.0,
                f"{partition_count} partitions, {total_samples:,} samples, {duration:.1f}s",
                fontsize=10,
                ha="center",
                color="gray",
                zorder=16,
            )

    # Draw partitions with better spacing and size info
    partition_y_positions = {}
    partition_count = len(timing_data["partitions"])

    for i, partition_id in enumerate(sorted(timing_data["partitions"].keys())):
        y_pos = 14.0 - (i * 1.2)  # Increased spacing between partitions
        partition_y_positions[partition_id] = y_pos

        # Get partition size info
        partition_info = timing_data["partitions"][partition_id]
        sample_count = 0
        if "creation_metadata" in partition_info:
            sample_count = partition_info["creation_metadata"].get("sample_count", 0)

        # Partition header with larger size and sample count
        partition_box = FancyBboxPatch(
            (0.5, y_pos - 0.4),
            2.0,
            0.8,
            boxstyle="round,pad=0.1",
            facecolor=colors["partition"],
            edgecolor="black",
            linewidth=2,
            zorder=12,
        )
        ax.add_patch(partition_box)
        ax.text(1.5, y_pos + 0.15, f"Partition {partition_id}", fontsize=12, fontweight="bold", ha="center", zorder=13)
        if sample_count > 0:
            ax.text(1.5, y_pos - 0.15, f"{sample_count:,} samples", fontsize=10, ha="center", color="gray", zorder=13)

    # Draw operations for each partition with better spacing
    operation_height = 0.5  # Increased operation height

    for op_key, op_data in timing_data["operations"].items():
        partition_id = op_data["partition_id"]
        op_name = op_data["op_name"]

        if partition_id in partition_y_positions:
            y_pos = partition_y_positions[partition_id]

            # Get color for specific operation
            op_color = colors.get(op_name, colors["clean_links_mapper"])

            # Calculate position
            if "start" in op_data and "end" in op_data:
                start_x = 3.0 + (op_data["start"] - base_time) * time_scale  # Moved right for more space
                end_x = 3.0 + (op_data["end"] - base_time) * time_scale
                width = end_x - start_x

                if width < 0.2:  # Increased minimum width for visibility
                    width = 0.2

                # Draw operation box
                op_box = FancyBboxPatch(
                    (start_x, y_pos - operation_height / 2),
                    width,
                    operation_height,
                    boxstyle="round,pad=0.05",
                    facecolor=op_color,
                    edgecolor="black",
                    linewidth=1,
                    zorder=5,
                )
                ax.add_patch(op_box)

                # Only show duration label (no operation name on bars)
                if "duration" in op_data:
                    duration = op_data["duration"]
                    ax.text(
                        start_x + width / 2,
                        y_pos,
                        f"{duration:.1f}s",
                        fontsize=9,
                        ha="center",
                        va="center",
                        color="gray",
                        fontweight="bold",
                        zorder=6,
                    )

                # Checkpoint indicator with better visibility
                if op_key in timing_data["checkpoints"]:
                    checkpoint_x = 3.0 + (timing_data["checkpoints"][op_key] - base_time) * time_scale
                    ax.plot(
                        [checkpoint_x, checkpoint_x],
                        [y_pos - 0.4, y_pos + 0.4],
                        color="orange",
                        linewidth=4,
                        alpha=0.8,
                        zorder=7,
                    )
                    ax.text(
                        checkpoint_x + 0.15, y_pos + 0.25, "CP", fontsize=8, color="orange", fontweight="bold", zorder=8
                    )

    # Draw merging phase - aligned with dataset partitioning at the top
    if all_times:
        max_time = max(all_times)
        merge_start = max_time + 5  # Assume merging starts 5 seconds after last operation
        merge_end = merge_start + 10  # Assume 10 seconds for merging

        start_x = 3.0 + (merge_start - base_time) * time_scale
        end_x = 3.0 + (merge_end - base_time) * time_scale
        width = end_x - start_x

        merge_box = FancyBboxPatch(
            (start_x, 15.0),
            width,
            0.6,
            boxstyle="round,pad=0.1",
            facecolor=colors["merging"],
            edgecolor="black",
            linewidth=2,
            zorder=15,
        )
        ax.add_patch(merge_box)
        ax.text(start_x + width / 2, 15.3, "Results Merging", fontsize=12, fontweight="bold", ha="center", zorder=16)
        ax.text(
            start_x + width / 2, 15.0, "Combining partition outputs", fontsize=10, ha="center", color="gray", zorder=16
        )

    # Draw timeline at the top with better positioning and high z-order
    ax.plot([3.0, 23.0], [16.5, 16.5], "k-", linewidth=3, zorder=20)

    # Timeline markers with better spacing and high z-order
    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        time_span = max_time - min_time

        for i in range(8):  # More timeline markers
            t = min_time + (i * time_span / 7)
            x = 3.0 + (t - base_time) * time_scale
            ax.plot([x, x], [16.2, 16.8], "k-", linewidth=2, zorder=20)
            time_str = f"{t - base_time:.0f}s"
            ax.text(x, 17.0, time_str, fontsize=10, ha="center", fontweight="bold", zorder=21)

    # Comprehensive legend system (positioned to avoid overlapping with main flow)
    legend_x = 0.5
    legend_y = 2.0

    # Operation types legend with specific colors
    ax.text(legend_x, legend_y + 1.0, "Operation Types & Colors:", fontsize=12, fontweight="bold", zorder=11)

    # Mapper operations
    ax.text(legend_x, legend_y + 0.6, "Mapper Operations:", fontsize=11, fontweight="bold", zorder=11)
    mapper_ops = [
        ("clean_links_mapper", "Clean Links"),
        ("clean_email_mapper", "Clean Email"),
        ("whitespace_normalization_mapper", "Whitespace Normalization"),
        ("fix_unicode_mapper", "Fix Unicode"),
    ]

    for i, (op_name, display_name) in enumerate(mapper_ops):
        legend_box = FancyBboxPatch(
            (legend_x, legend_y + 0.2 - i * 0.25),
            0.3,
            0.2,
            boxstyle="round,pad=0.05",
            facecolor=colors[op_name],
            edgecolor="black",
            zorder=10,
        )
        ax.add_patch(legend_box)
        ax.text(legend_x + 0.35, legend_y + 0.2 - i * 0.25 + 0.1, display_name, fontsize=10, zorder=11)

    # Filter operations
    filter_start_y = legend_y - 1.2
    ax.text(legend_x, filter_start_y + 0.6, "Filter Operations:", fontsize=11, fontweight="bold", zorder=11)
    filter_ops = [
        ("text_length_filter", "Text Length"),
        ("alphanumeric_filter", "Alphanumeric"),
        ("character_repetition_filter", "Character Repetition"),
        ("word_repetition_filter", "Word Repetition"),
    ]

    for i, (op_name, display_name) in enumerate(filter_ops):
        legend_box = FancyBboxPatch(
            (legend_x, filter_start_y + 0.2 - i * 0.25),
            0.3,
            0.2,
            boxstyle="round,pad=0.05",
            facecolor=colors[op_name],
            edgecolor="black",
            zorder=10,
        )
        ax.add_patch(legend_box)
        ax.text(legend_x + 0.35, filter_start_y + 0.2 - i * 0.25 + 0.1, display_name, fontsize=10, zorder=11)

    # System operations
    system_start_y = filter_start_y - 1.2
    ax.text(legend_x, system_start_y + 0.6, "System Operations:", fontsize=11, fontweight="bold", zorder=11)
    system_ops = [("partition_creation", "Partition Creation"), ("merging", "Results Merging")]

    for i, (op_name, display_name) in enumerate(system_ops):
        legend_box = FancyBboxPatch(
            (legend_x, system_start_y + 0.2 - i * 0.25),
            0.3,
            0.2,
            boxstyle="round,pad=0.05",
            facecolor=colors[op_name],
            edgecolor="black",
            zorder=10,
        )
        ax.add_patch(legend_box)
        ax.text(legend_x + 0.35, system_start_y + 0.2 - i * 0.25 + 0.1, display_name, fontsize=10, zorder=11)

    # Checkpoint legend
    checkpoint_y = system_start_y - 0.8
    ax.text(legend_x, checkpoint_y + 0.3, "Checkpoints:", fontsize=11, fontweight="bold", zorder=11)
    ax.plot([legend_x, legend_x + 0.3], [checkpoint_y, checkpoint_y], color="orange", linewidth=4, alpha=0.8, zorder=10)
    ax.text(legend_x + 0.35, checkpoint_y, "CP - Checkpoint Save", fontsize=10, zorder=11)

    plt.tight_layout()
    return fig


def main():
    """Main function to generate the execution graph."""
    parser = argparse.ArgumentParser(
        description="Generate execution graph from DataJuicer work directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python execution_graph_generator.py outputs/partition-checkpoint-eventlog/20250808_230030_501c9d
  python execution_graph_generator.py /path/to/your/work/directory
  python execution_graph_generator.py .  # Use current directory
        """,
    )
    parser.add_argument(
        "work_dir", help="Path to the DataJuicer work directory containing events.jsonl and dag_execution_plan.json"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="execution_graph.png",
        help="Output filename for the graph (default: execution_graph.png)",
    )

    args = parser.parse_args()

    # Validate work directory
    if not os.path.exists(args.work_dir):
        print(f"Error: Work directory '{args.work_dir}' does not exist")
        return 1

    try:
        # Find required files
        events_file, dag_file = find_files_in_work_dir(args.work_dir)

        if not events_file:
            print("Error: Could not find events.jsonl file")
            return 1

        if not dag_file:
            print("Error: Could not find dag_execution_plan.json file")
            return 1

        # Parse data
        events = parse_events(events_file)
        dag_plan = parse_dag_plan(dag_file)

        if not events:
            print("Error: No events found in events.jsonl file")
            return 1

        # Extract timing data
        timing_data = extract_timing_data(events)

        # Create visualization
        fig = create_execution_graph(timing_data, dag_plan)

        # Save the graph
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Execution graph saved to {args.output}")

        # Display summary
        print("\nExecution Summary:")
        print(f"Total partitions: {len(timing_data['partitions'])}")
        print(f"Total operations: {len(timing_data['operations'])}")
        print(f"Checkpoints created: {len(timing_data['checkpoints'])}")

        if timing_data["job_start"] and timing_data["job_end"]:
            duration = timing_data["job_end"] - timing_data["job_start"]
            print(f"Total job duration: {duration:.2f} seconds")

        # Close the figure to free memory
        plt.close(fig)
        print("\nGraph generation completed successfully!")
        return 0

    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return 1
    except Exception as e:
        print(f"Error generating execution graph: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
