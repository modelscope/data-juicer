import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from datasets.utils.extract import Extractor as HF_Extractor
from datasets.utils.filelock import FileLock as HF_FileLock
from loguru import logger


class FileLock(HF_FileLock):
    """
    File lock for compresssion or decompression, and
    remove lock file automatically.
    """

    def _release(self):
        super()._release()
        try:
            logger.info(f'Remove {self._lock_file}')
            os.remove(self._lock_file)
        # The file is already deleted and that's what we want.
        except OSError:
            pass
        return None


class Extractor(HF_Extractor):
    """
    Extract content from a compressed file.
    """

    @classmethod
    def extract(
        cls,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        extractor_format: str,
    ):
        """
        Extract content from a compressed file.
        :param input_path: path to compressed file.
        :param output_path: path to uncompressed file.
        :param extractor_format: extraction format,
            see supported algorithm in `Extractor` of huggingface dataset.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Prevent parallel extractions
        lock_path = str(Path(output_path).with_suffix('.lock'))
        with FileLock(lock_path):
            shutil.rmtree(output_path, ignore_errors=True)
            extractor = cls.extractors[extractor_format]
            return extractor.extract(input_path, output_path)


class BaseCompressor(ABC):
    """
    Base class that compresses a file.
    """

    @staticmethod
    @abstractmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress input file and save to output file.
        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        ...


class ZstdCompressor(BaseCompressor):
    """
    This class compresses a file using the `zstd` algorithm.
    """

    @staticmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress input file and save to output file.
        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """

        import zstandard as zstd

        cctx = zstd.ZstdCompressor()
        with open(input_path, 'rb') as ifh, open(output_path, 'wb') as ofh:
            cctx.copy_stream(ifh, ofh)


class Lz4Compressor(BaseCompressor):
    """
    This class compresses a file using the `lz4` algorithm.
    """

    @staticmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress a input file and save to output file.
        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        import lz4.frame

        with open(input_path, 'rb') as input_file:
            with lz4.frame.open(output_path, 'wb') as compressed_file:
                shutil.copyfileobj(input_file, compressed_file)


class GzipCompressor(BaseCompressor):
    """
    This class compresses a file using the `gzip` algorithm.
    """

    @staticmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress input file and save to output file.
        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        import gzip

        with open(input_path, 'rb') as input_file:
            with gzip.open(output_path, 'wb') as compressed_file:
                shutil.copyfileobj(input_file, compressed_file)


class Compressor:
    """
    This class that contains multiple compressors.
    """
    compressors: Dict[str, Type[BaseCompressor]] = {
        'gzip': GzipCompressor,
        # "zip": ZipCompressor,
        # "xz": XzCompressor,
        # "rar": RarCompressor,
        'zstd': ZstdCompressor,
        # "bz2": Bzip2Compressor,
        # "7z": SevenZipCompressor,
        'lz4': Lz4Compressor,
    }

    @classmethod
    def compress(
        cls,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        compressor_format: str,
    ):
        """
        Compress input file and save to output file.
        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        :param compressor_format: compression format,
            see supported algorithm in `compressors`.
        """

        assert compressor_format in cls.compressors.keys()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Prevent parallel extractions
        lock_path = str(Path(output_path).with_suffix('.lock'))
        with FileLock(lock_path):
            shutil.rmtree(output_path, ignore_errors=True)
            compressor = cls.compressors[compressor_format]
            compressor.compress(input_path, output_path)


class CompressManager:
    """
    This class is used to compress or decompress a input file
    using compression format algorithms.
    """

    def __init__(self, compressor_format: str = 'zstd'):
        """
        Initialization method.

        :param compressor_format: compression format algorithms,
            default `zstd`.
        """

        assert compressor_format in Compressor.compressors.keys()
        self.compressor_format = compressor_format
        self.compressor = Compressor
        self.extractor = Extractor

    def compress(
        self,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
    ):
        """
        Compress input file and save to output file.
        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        self.compressor.compress(input_path, output_path,
                                 self.compressor_format)

    def decompress(
        self,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
    ):
        """
        Decompress input file and save to output file.
        :param input_path: path to compressed file.
        :param output_path: path to uncompressed file.
        """
        self.extractor.extract(input_path, output_path, self.compressor_format)


class CacheCompressManager:
    """
    This class is used to compress or decompress huggingface cache files
    using compression format algorithms.
    """

    def __init__(self, compressor_format: str = 'zstd'):
        """
        Initialization method.

        :param compressor_format: compression format algorithms,
            default `zstd`.
        """
        self.compressor_format = compressor_format
        self.compressor_extension = '.' + compressor_format
        self.compress_manager = CompressManager(
            compressor_format=compressor_format)
        self.pattern = re.compile('_\d{5}_of_')  # noqa W605

    def _get_raw_filename(self, filename: Union[Path, str]):
        """
        Get a uncompressed file name from a compressed file.
        :param filename: path to compressed file.
        :return: path to uncompressed file.
        """
        assert filename.endswith(self.compressor_format)
        return str(filename)[:-len(self.compressor_extension)]

    def _get_compressed_filename(self, filename: Union[Path, str]):
        """
        Get a compressed file name from a uncompressed file.
        :param filename: path to uncompressed file.
        :return: path to compressed file.
        """
        return str(filename) + self.compressor_extension

    def _get_cache_diretory(self, ds) -> str:
        """
        Get dataset cache directory.
        :param ds: input dataset.
        :return: dataset cache directory.
        """
        current_cache_files = [
            os.path.abspath(cache_file['filename'])
            for cache_file in ds.cache_files
        ]
        if not current_cache_files:
            return None
        cache_directory = os.path.dirname(current_cache_files[0])
        return cache_directory

    def _get_cache_file_names(self,
                              cache_directory: str,
                              fingerprint: Optional[str] = None,
                              extension='.arrow'):
        """
        Get all cache files in the dataset cache directory with fingerprint,
        which ends with specified extension.
        :param cache_directory: dataset cache directory.
        :param fingerprint: fingerprint of cache files.
            If `None`, we will find all cache files which starts with
            `cache-` and ends with specified extension
        :param extension: extension of cache files, default `.arrow`
        :return: list of file names
        """
        if cache_directory is None:
            return []
        if fingerprint is None:
            fingerprint = ''

        files: List[str] = os.listdir(cache_directory)
        f_names = []
        for f_name in files:
            if f_name.startswith(f'cache-{fingerprint}') and f_name.endswith(
                    extension):
                f_names.append(f_name)
        return f_names

    def compress(self, ds, fingerprint: Optional[str] = None):
        """
        Compress cache files with fingerprint in dataset cache directory.
        :param ds: input dataset.
        :param fingerprint: fingerprint of cache files.
            If `None`, we will find all cache files which starts with
            `cache-` and ends with `.arrow`
        """
        cache_directory = self._get_cache_diretory(ds)
        if cache_directory is None:
            return

        f_names = self._get_cache_file_names(cache_directory=cache_directory,
                                             fingerprint=fingerprint,
                                             extension='.arrow')
        files_to_remove = []
        files_printed = set()
        for f_name in f_names:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            compress_filename = self._get_compressed_filename(full_name)
            formated_cache_name = self.format_cache_file_name(
                compress_filename)

            if not os.path.exists(compress_filename):
                if formated_cache_name not in files_printed:
                    logger.info(
                        f'Compress cache file to {formated_cache_name}')
                    self.compress_manager.compress(full_name,
                                                   compress_filename)
            else:
                if formated_cache_name not in files_printed:
                    logger.info(
                        f'Found compressed cache file {formated_cache_name}')
            files_printed.add(formated_cache_name)
            files_to_remove.append(full_name)

        # clean up raw cache file
        for file_path in files_to_remove:
            logger.debug(f'Removing cache file {file_path}')
            os.remove(file_path)

    def decompress(self, ds, fingerprint: Optional[str] = None):
        """
        Decompress compressed cache files with fingerprint in
        dataset cache directory.
        :param ds: input dataset.
        :param fingerprint: fingerprint of cache files.
            If `None`, we will find all cache files which starts with
            `cache-` and ends with compression format.
        """
        cache_directory = self._get_cache_diretory(ds)
        if cache_directory is None:
            return

        f_names = self._get_cache_file_names(
            cache_directory=cache_directory,
            fingerprint=fingerprint,
            extension=self.compressor_extension)
        files_printed = set()
        for f_name in f_names:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            raw_filename = self._get_raw_filename(full_name)
            formated_cache_name = self.format_cache_file_name(raw_filename)

            if formated_cache_name not in files_printed:
                logger.info(f'Decompress cache file to {formated_cache_name}')
                files_printed.add(formated_cache_name)
            self.compress_manager.decompress(full_name, raw_filename)

    def format_cache_file_name(
            self, cache_file_name: Optional[str]) -> Optional[str]:
        """
        Use `*` to replace the sub rank in a cache file name.
        :param cache_file_name: a cache file name.
        """

        if not cache_file_name:
            return cache_file_name

        cache_file_name = re.sub(pattern=self.pattern,
                                 repl=r'_*_of_',
                                 string=cache_file_name)
        return cache_file_name

    def cleanup_cache_files(self, ds):
        """
        Clean up all compressed cache files in dataset cache directory,
        which starts with `cache-` and ends with compression format
        :param ds: input dataset.
        """
        cache_directory = self._get_cache_diretory(ds)
        if cache_directory is None:
            return
        f_names = self._get_cache_file_names(
            cache_directory=cache_directory,
            extension=self.compressor_extension)
        files_printed = set()
        for f_name in f_names:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            formated_cache_name = self.format_cache_file_name(full_name)
            if formated_cache_name not in files_printed:
                logger.debug(f'Clean up cache file {formated_cache_name}')
                files_printed.add(formated_cache_name)
            os.remove(full_name)
        return len(f_names)
