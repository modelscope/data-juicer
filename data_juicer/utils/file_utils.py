from pathlib import Path
from typing import List, Tuple, Union

from datasets.utils.extract import ZstdExtractor as Extractor


def find_files_with_suffix(
        path: Union[str, Path],
        suffixes: Union[str, List[str], Tuple[str]] = None) -> List[str]:
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
