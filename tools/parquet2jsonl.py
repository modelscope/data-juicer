import glob
import os
from typing import List

import pandas as pd
import pyarrow.parquet as pq
import ray


def parquet_to_jsonl(input_file: str, output_file: str) -> None:
    """
    Convert Parquet file to JSONL format.
    """
    with pq.ParquetFile(input_file) as parquet_file, \
            open(output_file, 'w', encoding='utf-8') as output:
        total_rows_processed = sum(
            output.write(
                parquet_file.read_row_group(i).to_pandas().to_json(
                    orient='records', lines=True, force_ascii=True))
            or parquet_file.read_row_group(i).num_rows
            for i in range(parquet_file.num_row_groups))
    print(f'Conversion complete. Total rows processed: {total_rows_processed}')


def get_parquet_file_names(dataset_dir_path: str) -> List[str]:
    """
    Load all parquet files in a directory.
    """
    if os.path.isdir(dataset_dir_path):
        parquet_files = glob.glob(os.path.join(dataset_dir_path, '*.parquet'))
    elif os.path.isfile(dataset_dir_path) and dataset_dir_path.lower(
    ).endswith('.parquet'):
        parquet_files = [dataset_dir_path]
    else:
        raise ValueError(
            'Invalid path: it should be a directory containing parquet files'
            ' or a single parquet file.')
    return parquet_files


def convert_parquet_to_jsonl(parquet_path: str, output_dir: str):
    """Convert a parquet file to jsonl format."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.basename(parquet_path)
    output_file = os.path.join(output_dir,
                               base_name.replace('.parquet', '.jsonl'))

    parquet_to_jsonl(parquet_path, output_file)
    print(f'Converted {parquet_path} to {output_file}')


def main(args):
    ray.init(args.ray_address)
    data_dir = args.data_dir
    parquet_files = get_parquet_file_names(data_dir)
    df = pd.DataFrame({'parquet_files': parquet_files})
    data = ray.data.from_pandas(df)

    def process_parquet(parquet_paths: pd.DataFrame) -> List[str]:
        for parquet_path in parquet_paths['parquet_files']:
            convert_parquet_to_jsonl(parquet_path, args.output_dir)
        return parquet_paths

    data.map_batches(process_parquet).materialize()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ray-address',
                        type=str,
                        default='auto',
                        help='The address of the Ray cluster.')
    parser.add_argument('--data-dir',
                        '-i',
                        type=str,
                        required=True,
                        help='Path to your dataset(parquet) directory.')
    parser.add_argument('--output-dir',
                        type=str,
                        required=True,
                        help='Path to your output(jsonl) directory.')
    args = parser.parse_args()
    main(args)
