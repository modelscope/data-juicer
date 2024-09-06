from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Generator, List, Union

import pandas as pd
import pyarrow as pa
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.ops import Filter, Mapper
from data_juicer.ops.base_op import OP
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import DEFAULT_MAX_FILE_SIZE, Fields
from data_juicer.utils.process_utils import calculate_np

with AvailabilityChecking(['ray'], requires_type='dist'):
    import ray
    import ray.data as rd
    from ray.data import Dataset


def is_valid_path(item, dataset_dir):
    full_path = os.path.abspath(os.path.join(dataset_dir, item))
    return os.path.exists(full_path)


def convert_to_absolute_paths(dict_with_paths, dataset_dir, path_keys):
    for key in path_keys:
        if key not in dict_with_paths:
            continue
        if isinstance(dict_with_paths[key], list):
            dict_with_paths[key] = [
                os.path.abspath(os.path.join(dataset_dir, item))
                if isinstance(item, str) and is_valid_path(dataset_dir, item)
                else item for item in dict_with_paths[key]
            ]
        elif isinstance(dict_with_paths[key], str):
            dict_with_paths[key] = os.path.abspath(
                os.path.join(dataset_dir,
                             dict_with_paths[key])) if is_valid_path(
                                 dict_with_paths[key],
                                 dataset_dir) else dict_with_paths[key]
    return dict_with_paths


class RayPreprocessOperator(OP):

    def __init__(self, dataset_path=None, cfg=None) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.cfg = cfg
        self._name = 'RayPreporcess'

    def run(self, dataset: Dataset) -> Dataset:
        columns = dataset.columns()
        if Fields.stats not in columns:
            logger.info(f'columns {columns}')

            def process_batch_arrow(table: pa.Table) -> pa.Table:
                new_column_data = [{} for _ in range(len(table))]
                new_talbe = table.append_column(Fields.stats,
                                                [new_column_data])
                return new_talbe

            dataset = dataset.map_batches(process_batch_arrow,
                                          batch_format='pyarrow')
        if self.dataset_path:
            # TODO: check path for nestdataset
            if not (self.cfg.video_key in dataset.columns()
                    or self.cfg.image_key in dataset.columns()
                    or self.cfg.audio_key in dataset.columns()):
                return dataset
            dataset_dir = os.path.dirname(self.dataset_path)
            dataset = dataset.map(lambda item: convert_to_absolute_paths(
                item, dataset_dir,
                [self.cfg.video_key, self.cfg.image_key, self.cfg.audio_key]))
        return dataset

    def use_cuda(self):
        return False


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu


def split_jsonl(file_path: str, max_size: int,
                output_dir: str) -> Generator[str]:
    """Split a jsonl file into multiple sub files more efficiently.

    Args:
        file_path (`str`): path of the original jsonl file
        max_size (`int`): max size of each sub file (in MB)
        output_dir (`str`): directory to save the sub files

    Yields:
        str: path of each newly created sub file
    """
    os.makedirs(output_dir, exist_ok=True)
    file_index = 0
    max_byte_size = max_size * 1024**2
    base_file_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_file_name)[0]
    current_size = 0
    buffer = []
    buffer_size = 0
    logger.info(f'Spliting {file_path}.')

    with open(file_path, 'r', encoding='utf-8') as infile:
        while True:
            # Determine the output file name
            output_file_name = f'{file_name}_{file_index}.jsonl'
            output_file_path = os.path.join(output_dir, output_file_name)

            # Read lines until we reach the max buffer size
            while current_size + buffer_size < max_byte_size:
                line = infile.readline()
                if not line:
                    break
                buffer.append(line)
                buffer_size += len(line)

            # Write the buffered lines to the current output file
            if buffer:
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.writelines(buffer)
                buffer = []
                buffer_size = 0
                file_index += 1
                yield output_file_path

            if not line:
                break


def parallel_split_jsonl(file_path, max_size, output_dir) -> List[str]:
    """Wrapper function for using with ThreadPoolExecutor."""
    return list(split_jsonl(file_path, max_size, output_dir))


def split_jsonl_dataset(
    dataset_paths: Union[str, List[str]],
    max_size: int,
    output_dir: str,
) -> Generator[str]:
    """Re-split the jsonl dataset files.

    Args:
        file_path (`str`): path of the original jsonl file
        max_size (`int`): max size of each sub file (in MB)
        output_dir (`str`): directory to save the sub files

    Yields:
        str: path of each newly created sub file
    """
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    logger.info('Re-splitting dataset files...')

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(parallel_split_jsonl, path, max_size, output_dir):
            path
            for path in dataset_paths
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                for result in results:
                    logger.info(f'Splited into {result}')
                    yield result
            except Exception as e:
                logger.error(f'Failed to split file: {e}')


def get_jsonl_file_names(dataset_dir_path: str) -> List[str]:
    """Load all jsonl files in a directory.

    Args:
        dataset_dir_path (`str`): path of the directory containing jsonl files
        or the path of a single jsonl file

    Returns:
        List[str]: list of jsonl file paths
    """
    if os.path.isdir(dataset_dir_path):
        jsonl_files = [
            os.path.join(dataset_dir_path, f)
            for f in os.listdir(dataset_dir_path)
        ]
    elif os.path.isfile(dataset_dir_path) and dataset_dir_path.endswith(
            '.jsonl') or dataset_dir_path.endswith('.json'):
        jsonl_files = [dataset_dir_path]
    else:
        raise ValueError(
            'Invalid path: it should be a directory containing jsonl files'
            ' or a single jsonl file.')
    return jsonl_files


def best_file_num(cpu: int, memory: int, file_size: int) -> int:
    """Calculate the best number of files in a single batch.
    The total memory should be at least 4 times larger than the
    total file size.

    Args:
        cpu (`int`): number of CPUs available
        memory (`int`): memory available in MB
        file_size (`int`): size of a single file in MB

    Returns:
        int: best number of files in a single batch
    """
    max_files_by_memory = memory // (4 * file_size)
    best_num_files = min(cpu, max_files_by_memory)
    logger.info(f'Best number of files in a single batch: {best_num_files}')
    return best_num_files


def load_splited_json_dataset(
    file_paths: Generator[str],
    file_num_in_batch: int,
) -> Generator[Dataset, None, None]:
    """Load dataset from the splited jsonl files.

    Args:
        file_paths (`Generator[str]`):
            A list of paths to the JSONL files.
        file_num_in_batch (`int`):
            The number of files to be included in each batch.

    Yields:
        `Dataset`: A dataset containing data from the specified batch of files.
    """
    files = []
    for file_path in file_paths:
        files.append(file_path)
        if len(files) >= file_num_in_batch:
            yield rd.read_json(files)
            files.clear()
    if len(files) > 0:
        yield rd.read_json(files)


class RayDataset(DJDataset):

    def __init__(self, datasets: Union[Dataset, Generator], cfg=None) -> None:
        self.cfg = cfg
        self.num_proc = None
        if isinstance(datasets, Dataset):
            self.datasets = [datasets]
        else:
            self.datasets = datasets
        if cfg:
            self.num_proc = cfg.np
        self.output_dataset = []
        self._ops = [RayPreprocessOperator(dataset_path=None, cfg=cfg)]

    @classmethod
    def read_jsonl(cls,
                   path: Union[str, List[str]],
                   cfg: Any = None,
                   resplit: bool = True) -> RayDataset:
        if resplit:
            resplit_dir = os.path.join(cfg.work_dir, 'resplit')
            os.makedirs(resplit_dir, exist_ok=True)
            files = split_jsonl_dataset(get_jsonl_file_names(path),
                                        DEFAULT_MAX_FILE_SIZE, resplit_dir)
            cpu = ray.cluster_resources().get('CPU', 0)
            memory = ray.cluster_resources().get('memory', 0) / 1024 / 1024
            logger.info(f'CPU: {cpu}, Memory: {memory}')
            batch_file_num = best_file_num(cpu, memory, DEFAULT_MAX_FILE_SIZE)
            return RayDataset(datasets=load_splited_json_dataset(
                files, batch_file_num),
                              cfg=cfg)
        else:
            return RayDataset(datasets=rd.read_json(path), cfg=cfg)

    @classmethod
    def read_item(cls, data: dict, cfg: Any = None) -> RayDataset:
        return RayDataset(dataset=rd.from_items(data), cfg=cfg)

    def process(self, operators) -> DJDataset:
        if not isinstance(operators, list):
            operators = [operators]
        self._ops.extend(operators)
        return self

    def _run_single_op(self, op, dataset: Dataset) -> Dataset:
        op_proc = calculate_np(op._name, op.mem_required, op.cpu_required,
                               self.num_proc, op.use_cuda())
        num_gpus = get_num_gpus(op, op_proc)
        try:
            if isinstance(op, Mapper):
                dataset = dataset.map_batches(op.process,
                                              batch_size=1,
                                              batch_format='pyarrow',
                                              num_gpus=num_gpus)
            elif isinstance(op, Filter):
                dataset = dataset.map_batches(op.compute_stats,
                                              batch_size=1,
                                              batch_format='pyarrow',
                                              num_gpus=num_gpus)
                if op.stats_export_path is not None:
                    dataset.write_json(op.stats_export_path, force_ascii=False)
                dataset = dataset.filter(op.process)
            elif isinstance(op, RayPreprocessOperator):
                dataset = op.run(dataset)
            else:
                logger.error(
                    'Ray executor only support Filter and Mapper OPs for now')
                raise NotImplementedError
            return dataset
        except:  # noqa: E722
            logger.error(f'An error occurred during Op [{op._name}].')
            import traceback
            traceback.print_exc()
            exit(1)

    def to_pandas(self) -> pd.DataFrame:
        dfs = []
        for data in self.datasets:
            dfs.append(data.to_pandas())
        return pd.concat(dfs, ignore_index=True)

    def write_json(self, path: str, force_ascii: bool = False) -> None:
        for dataset in self.datasets:
            if len(self._ops) > 0:
                for op in self._ops:
                    dataset = self._run_single_op(op, dataset)
                dataset.write_json(path, force_ascii=force_ascii)
