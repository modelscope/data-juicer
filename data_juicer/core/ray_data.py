from __future__ import annotations

import os
from typing import Any, Generator, List, Union

import pyarrow as pa
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.core.data import DJDataset
from data_juicer.ops import Filter, Mapper
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


# TODO: check path for nestdataset
def set_dataset_to_absolute_path(dataset, dataset_path, cfg):
    """
    Set all the path in input data to absolute path.
    Checks dataset_dir and project_dir for valid paths.
    """
    if not (cfg.video_key in dataset.columns() or cfg.image_key
            in dataset.columns() or cfg.audio_key in dataset.columns()):
        return dataset
    dataset_dir = os.path.dirname(dataset_path)
    dataset = dataset.map(lambda item: convert_to_absolute_paths(
        item, dataset_dir, [cfg.video_key, cfg.image_key, cfg.audio_key]))
    logger.info(f"transfer {dataset.count()} sample's paths")
    return dataset


def preprocess_dataset(dataset: Dataset, dataset_path, cfg) -> Dataset:
    if dataset_path:
        dataset = set_dataset_to_absolute_path(dataset, dataset_path, cfg)
    columns = dataset.columns()
    if Fields.stats not in columns:
        logger.info(f'columns {columns}')

        def process_batch_arrow(table: pa.Table) -> pa.Table:
            new_column_data = [{} for _ in range(len(table))]
            new_talbe = table.append_column(Fields.stats, [new_column_data])
            return new_talbe

        dataset = dataset.map_batches(process_batch_arrow,
                                      batch_format='pyarrow')
    return dataset


def get_num_gpus(op, op_proc):
    if not op.use_cuda():
        return 0
    proc_per_gpu = op_proc / cuda_device_count()
    return 1.0 / proc_per_gpu


def split_jsonl(file_path: str, max_size: int, output_dir: str) -> List[str]:
    """Split a jsonl file into multiple sub files

    Args:
        file_path (`str`): path of the original jsonl file
        max_size (`int`): max size of each sub file (in MB)
        output_dir (`str`): directory to save the sub files

    Returns:
        List[str]: list of sub file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    sub_file_paths = []
    file_index = 0
    max_byte_size = max_size * 1024**2
    base_file_name = os.path.basename(file_path)
    file_name = os.path.splitext(base_file_name)[0]
    current_size = 0
    with open(file_path, 'r', encoding='utf-8') as infile:
        while True:
            # Create a new file if we're starting or have reached the max size
            if current_size >= max_byte_size:
                file_index += 1
                current_size = 0
            # Determine the output file name
            output_file_name = f'{file_name}_{file_index}.jsonl'
            output_file_path = os.path.join(output_dir, output_file_name)
            # Open the output file in append mode
            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                line = infile.readline()
                if not line:
                    break
                # Write the line to the current output file
                outfile.write(line)
                # Update the current file size
                current_size += len(line.encode('utf-8'))
            # Add the new file path to the list if it's a new file
            if output_file_path not in sub_file_paths:
                sub_file_paths.append(output_file_path)

    return sub_file_paths


def split_jsonl_dataset(
    dataset_paths: Union[str, List[str]],
    max_size: int,
    output_dir: str,
) -> List[str]:
    """Re-split the jsonl dataset files.

    Args:
        file_path (`str`): path of the original jsonl file
        max_size (`int`): max size of each sub file (in MB)
        output_dir (`str`): directory to save the sub files

    Returns:
        List[str]: list of sub file paths
    """
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    all_sub_file_paths = []
    for path in dataset_paths:
        sub_file_paths = split_jsonl(path, max_size, output_dir)
        all_sub_file_paths.extend(sub_file_paths)

    return all_sub_file_paths


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
    Each cpu should process the same number of files (at least one),
    while the total memory should be at least 2 times larger than the
    total file size.

    Args:
        cpu (`int`): number of CPUs available
        memory (`int`): memory available in MB
        file_size (`int`): size of a single file in MB

    Returns:
        int: best number of files in a single batch
    """
    max_files_by_memory = memory // (2 * file_size)

    best_num_files = max(1, (max_files_by_memory // cpu) * cpu)

    return best_num_files


def load_splited_json_dataset(
    file_paths: List[str],
    file_num_in_batch: int,
) -> Generator[Dataset, None, None]:
    """Load dataset from the splited jsonl files.

    Args:
        file_paths (`List[str]`):
            A list of paths to the JSONL files.
        file_num_in_batch (`int`):
            The number of files to be included in each batch.

    Yields:
        `Dataset`: A dataset containing data from the specified batch of files.
    """
    num_batches = (len(file_paths) + file_num_in_batch - 1)
    for i in range(num_batches):
        start_idx = i * file_num_in_batch
        end_idx = min((i + 1) * file_num_in_batch, len(file_paths))
        batch_file_paths = file_paths[start_idx:end_idx]
        dataset = rd.read_json(batch_file_paths)
        yield dataset


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

    @classmethod
    def read_jsonl(cls,
                   path: Union[str, List[str]],
                   cfg: Any = None) -> RayDataset:
        files = split_jsonl_dataset(get_jsonl_file_names(path),
                                    DEFAULT_MAX_FILE_SIZE, cfg.work_dir)
        cpu = ray.cluster_resources().get('CPU', 0)
        memory = ray.cluster_resources().get('memory', 0) / 1024 / 1024
        batch_file_num = best_file_num(cpu, memory, DEFAULT_MAX_FILE_SIZE)
        return RayDataset(dataset=load_splited_json_dataset(
            files, batch_file_num),
                          cfg=cfg)

    @classmethod
    def read_item(cls, data: dict, cfg: Any = None) -> RayDataset:
        return RayDataset(dataset=rd.from_items(data), cfg=cfg)

    def process(self,
                operators,
                *,
                exporter=None,
                checkpointer=None,
                tracer=None) -> DJDataset:
        outputs = []
        for dataset in self.datasets:
            # todo: pass dataset path into the function
            data = preprocess_dataset(dataset, self.cfg)
            if operators is None:
                return self
            if not isinstance(operators, list):
                operators = [operators]
            for op in operators:
                data = self._run_single_op(op, data)
            outputs.append(data)
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
