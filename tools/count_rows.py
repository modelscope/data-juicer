#!/usr/bin/env python3
"""
Different ways to count rows in a parquet file
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def get_parquet_info(file_path):
    """Get detailed information about the parquet file"""
    print(f"\nParquet file information for: {file_path}")
    print("-" * 50)

    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata

    print(f"Total rows: {metadata.num_rows:,}")
    print(f"Total columns: {metadata.num_columns}")
    print(f"Number of row groups: {metadata.num_row_groups}")
    print(f"File size: {metadata.serialized_size / 1024 / 1024:.2f} MB")

    # Show column information
    print("\nColumns:")
    for i in range(metadata.num_columns):
        col_meta = metadata.row_group(0).column(i)
        print(f"  {col_meta.path_in_schema}: {col_meta.physical_type}")


def count_rows_auto(file_path):
    """Automatically choose the best method based on file extension and count rows"""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    if extension == ".parquet":
        # Use pyarrow metadata for parquet - fastest and most efficient
        parquet_file = pq.ParquetFile(file_path)
        row_count = parquet_file.metadata.num_rows
        method_used = "pyarrow metadata"
    elif extension in [".csv", ".tsv"]:
        # For CSV files, use pandas
        df = pd.read_csv(file_path)
        row_count = len(df)
        method_used = "pandas read_csv"
    elif extension in [".json", ".jsonl"]:
        # For JSON files, use pandas
        if extension == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        else:
            df = pd.read_json(file_path)
        row_count = len(df)
        method_used = "pandas read_json"
    elif extension in [".arrow", ".feather"]:
        # For Arrow files, use pyarrow
        table = pa.ipc.open_file(file_path).read_all()
        row_count = table.num_rows
        method_used = "pyarrow arrow"
    else:
        # Default to pandas for unknown extensions
        try:
            df = pd.read_csv(file_path)
            row_count = len(df)
            method_used = "pandas read_csv (default)"
        except Exception as e:
            print(f"Error: Could not read file with extension {extension}: {e}")
            return None, None

    return row_count, method_used


def get_supported_extensions():
    """Return list of supported file extensions"""
    return [".parquet", ".csv", ".tsv", ".json", ".jsonl", ".arrow", ".feather"]


def count_directory(directory_path, show_info=False):
    """Count rows for all supported files in a directory"""
    directory_path = Path(directory_path)
    supported_extensions = get_supported_extensions()

    # Find all supported files in directory (recursive)
    files = []
    for ext in supported_extensions:
        files.extend(directory_path.rglob(f"*{ext}"))

    if not files:
        print(f"No supported files found in directory: {directory_path}")
        return

    # Sort files for consistent output
    files = sorted(files)

    print(f"Found {len(files)} supported files in: {directory_path}")
    print("=" * 80)

    total_rows = 0
    file_counts = []

    for file_path in files:
        try:
            row_count, method_used = count_rows_auto(file_path)
            if row_count is not None:
                file_counts.append(
                    {
                        "file": file_path,
                        "rows": row_count,
                        "method": method_used,
                        "size_mb": file_path.stat().st_size / 1024 / 1024,
                    }
                )
                total_rows += row_count
                print(f"{file_path.name:<50} {row_count:>10,} rows ({method_used})")
            else:
                print(f"{file_path.name:<50} {'ERROR':>10}")
        except Exception as e:
            print(f"{file_path.name:<50} {'ERROR':>10} - {e}")

    # Print summary
    print("=" * 80)
    print(f"Total files: {len(file_counts)}")
    print(f"Total rows: {total_rows:,}")
    print(f"Average rows per file: {total_rows // len(file_counts):,}")

    # Show detailed info for parquet files if requested
    if show_info:
        parquet_files = [f for f in file_counts if f["file"].suffix.lower() == ".parquet"]
        if parquet_files:
            print("\n" + "=" * 80)
            print("DETAILED PARQUET FILE INFORMATION")
            print("=" * 80)
            for file_info in parquet_files:
                get_parquet_info(file_info["file"])
                print()

    return file_counts, total_rows


def main():
    parser = argparse.ArgumentParser(description="Count rows in data files using the most appropriate method")
    parser.add_argument("--file", "-f", help="Path to a single data file")
    parser.add_argument("--directory", "-d", help="Path to a directory containing data files")
    parser.add_argument("--info", "-i", action="store_true", help="Show detailed file information (for parquet files)")

    args = parser.parse_args()

    # Check if either file or directory is provided
    if not args.file and not args.directory:
        parser.error("Either --file or --directory must be specified")

    if args.file and args.directory:
        parser.error("Cannot specify both --file and --directory")

    if args.file:
        # Single file mode
        file_path = args.file

        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return 1

        print(f"Counting rows in: {file_path}")
        print("=" * 60)

        row_count, method_used = count_rows_auto(file_path)

        if row_count is not None:
            print(f"Row count: {row_count:,}")
            print(f"Method used: {method_used}")
        else:
            return 1

        # Show detailed info for parquet files if requested
        if args.info and Path(file_path).suffix.lower() == ".parquet":
            get_parquet_info(file_path)

    elif args.directory:
        # Directory mode
        directory_path = args.directory

        if not Path(directory_path).exists():
            print(f"Error: Directory not found: {directory_path}")
            return 1

        if not Path(directory_path).is_dir():
            print(f"Error: Path is not a directory: {directory_path}")
            return 1

        count_directory(directory_path, show_info=args.info)

    return 0


if __name__ == "__main__":
    exit(main())
