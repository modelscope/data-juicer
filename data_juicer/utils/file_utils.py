import asyncio
import copy
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Union

from datasets.utils.extract import ZstdExtractor as Extractor

from data_juicer.utils.common_utils import dict_to_hash
from data_juicer.utils.constant import DEFAULT_PREFIX, Fields


async def follow_read(
    logfile_path: str,
    skip_existing_content: bool = False,
) -> AsyncGenerator:
    """Read a file in online and iterative manner

    Args:
        logfile_path (`str`):
            The file path to be read.
        skip_existing_content (`bool`, defaults to `False):
            If True, read from the end, otherwise read from the beginning.

    Returns:
        One line string of the file content.
    """
    # in most unix file systems, the read operation is safe
    # for a file being target file of another "write process"
    with open(logfile_path, 'r', encoding='utf-8', errors='ignore') as logfile:
        if skip_existing_content:
            # move to the file's end, similar to `tail -f`
            logfile.seek(0, 2)

        while True:
            line = logfile.readline()
            if not line:
                # no new line, wait to avoid CPU override
                await asyncio.sleep(0.1)
                continue
            yield line


def find_files_with_suffix(
        path: Union[str, Path],
        suffixes: Union[str, List[str], None] = None) -> List[str]:
    """
    Traverse a path to find all files with the specified suffixes.

    :param path: path (str/Path): source path
    :param suffixes: specified file suffixes, '.txt' or ['.txt', '.md']
        etc
    :return: list of all files with the specified suffixes
    """
    path = Path(path)
    file_dict = {}

    if suffixes is None:
        suffixes = []

    if isinstance(suffixes, str):
        suffixes = [suffixes]

    suffixes = [
        x.lower() if x.startswith('.') else '.' + x.lower() for x in suffixes
    ]

    if path.is_file():
        files = [path]
    else:
        searched_files = path.rglob('*')
        files = [file for file in searched_files if file.is_file()]

    extractor = Extractor

    # only keep the file with the specified suffixes
    for file in files:
        suffix = file.suffix.lower()

        if extractor.is_extractable(file):

            # TODO
            # hard code
            # only support zstd-format file now,
            # and use the last 2 sub-suffixes as the final suffix
            # just like '.jsonl.zst'
            file_suffixes = [suffix.lower() for suffix in file.suffixes]
            suffix = ''.join(file_suffixes[-2:])

        if not suffixes or (suffix in suffixes):
            if suffix not in file_dict:
                file_dict[suffix] = [str(file)]
            else:
                file_dict[suffix].append(str(file))
    return file_dict


def is_absolute_path(path: Union[str, Path]) -> bool:
    """
    Check whether input path is a absolute path.

    :param path: input path
    :return: True means input path is absolute path, False means input
        path is a relative path.
    """
    return Path(path).is_absolute()


def add_suffix_to_filename(filename, suffix):
    """
    Add a suffix to the filename. Only regard the content after the last dot
    as the file extension.
    E.g.
    1. abc.jpg + "_resized" --> abc_resized.jpg
    2. edf.xyz.csv + "_processed" --> edf.xyz_processed.csv
    3. /path/to/file.json + "_suf" --> /path/to/file_suf.json
    4. ds.tar.gz + "_whoops" --> ds.tar_whoops.gz (maybe unexpected)

    :param filename: input filename
    :param suffix: suffix string to be added
    """
    name, ext = os.path.splitext(filename)
    new_name = f'{name}{suffix}{ext}'
    return new_name


def create_directory_if_not_exists(directory_path):
    """
        create a directory if not exists, this function is process safe

        :param directory_path: directory path to be create
    """
    directory_path = os.path.abspath(directory_path)
    try:
        os.makedirs(directory_path, exist_ok=True)
    except FileExistsError:
        # We ignore the except from multi processes or threads.
        # Just make sure the directory exists.
        pass


def transfer_filename(original_filepath: Union[str, Path], op_name,
                      **op_kwargs):
    """
        According to the op and hashing its parameters 'op_kwargs' addition
        to the process id and current time as the 'hash_val', map the
        original_filepath to another unique file path. E.g.

            1. abc.jpg -->
                __dj__produced_data__/{op_name}/
                abc__dj_hash_#{hash_val}#.jpg
            2. ./abc.jpg -->
                ./__dj__produced_data__/{op_name}/
                abc__dj_hash_#{hash_val}#.jpg
            3. /path/to/abc.jpg -->
                /path/to/__dj__produced_data__/{op_name}/
                abc__dj_hash_#{hash_val}#.jpg
            4. /path/to/__dj__produced_data__/{op_name}/
                abc__dj_hash_#{hash_val1}#.jpg -->
                /path/to/__dj__produced_data__/{op_name}/
                abc__dj_hash_#{hash_val2}#.jpg

    """
    # produce the directory
    original_dir = os.path.dirname(original_filepath)
    dir_token = f'/{Fields.multimodal_data_output_dir}/'
    if dir_token in original_dir:
        original_dir = original_dir.split(dir_token)[0]
    new_dir = os.path.join(original_dir,
                           f'{Fields.multimodal_data_output_dir}/{op_name}')
    create_directory_if_not_exists(new_dir)

    # produce the unique hash code
    unique_parameters = copy.deepcopy(op_kwargs)
    unique_parameters[f'{DEFAULT_PREFIX}pid'] = os.getpid()
    unique_parameters[f'{DEFAULT_PREFIX}timestamp'] = str(
        datetime.now(timezone.utc))
    unique_hash = dict_to_hash(unique_parameters)

    # if the input data is produced by data-juicer, replace the hash code
    # else append hash value to filename
    def add_hash_value(text, new_hash_value):
        pattern = r'__dj_hash_#(.*?)#'

        match = re.search(pattern, text)
        # draw the string produced by data-juicer
        if match:
            text = text[:match.start()]

        return f'{text}__dj_hash_#{new_hash_value}#'

    original_filename = os.path.basename(original_filepath)
    name, ext = os.path.splitext(original_filename)
    new_name = add_hash_value(name, unique_hash)
    new_filepath = os.path.join(new_dir, f'{new_name}{ext}')

    return new_filepath


def copy_data(from_dir, to_dir, data_path):
    """
        Copy data from from_dir/data_path to to_dir/data_path.
        Return True if success.
    """
    from_path = os.path.join(from_dir, data_path)
    to_path = os.path.join(to_dir, data_path)
    if not os.path.exists(from_path):
        return False
    parent_dir = os.path.dirname(to_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    shutil.copy2(from_path, to_path)
    return True
