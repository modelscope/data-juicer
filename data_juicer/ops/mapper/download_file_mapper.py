import copy
import os
from typing import List, Union

from loguru import logger

from data_juicer.utils.file_utils import (download_files_parallel,
                                          is_remote_path)

from ..base_op import OPERATORS, Mapper

OP_NAME = 'download_file_mapper'


@OPERATORS.register_module(OP_NAME)
class DownloadFileMapper(Mapper):
    """Mapper to download url files to local files.
    """

    _batched_op = True

    def __init__(self,
                 save_dir: str = None,
                 download_field: str = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 retry_delay: int = 1,
                 max_delay: int = 60,
                 stream: bool = False,
                 headers=None,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param save_dir: The directory to save downloaded files.
        :param download_field: The filed name to get the url to download.
        :param max_retries: The maximum number of retries in case of errors.
        :param timeout: The timeout in seconds for each HTTP request.
        :param retry_delay: The delay between retries in seconds, exponential backoff with jitter.
        :param max_delay: The maximum delay between retries in seconds.
        :param stream: If True, the file will be downloaded in chunks.
            If False, the entire file will be downloaded at once.
        :param headers: The headers to include in the HTTP request.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.download_field = download_field
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_delay = max_delay
        self.stream = stream
        self.headers = headers

    def download_nested_urls(self, nested_urls: List[Union[str, List[str]]],
                             save_dir: str):
        flat_urls = []
        structure_info = []  # save as original index, sub index

        for idx, urls in enumerate(nested_urls):
            if isinstance(urls, list):
                for sub_idx, url in enumerate(urls):
                    if is_remote_path(url):
                        flat_urls.append(url)
                        structure_info.append((idx, sub_idx))
            else:
                if is_remote_path(urls):
                    flat_urls.append(urls)
                    structure_info.append(
                        (idx, -1))  # -1 means single str element

        download_results = download_files_parallel(
            flat_urls,
            save_dir,
            stream=self.stream,
            headers=self.headers,
            max_retries=self.max_retries,
            timeout=self.timeout,
            retry_delay=self.retry_delay,
            max_delay=self.max_delay)

        keep_failed_url = True
        if keep_failed_url:
            reconstructed = copy.deepcopy(nested_urls)
        else:
            reconstructed = []
            for item in nested_urls:
                if isinstance(item, list):
                    reconstructed.append([None] * len(item))
                else:
                    reconstructed.append(None)

        failed_info = ''
        for i, (success, save_path, response) in enumerate(download_results):
            orig_idx, sub_idx = structure_info[i]
            if not success:
                save_path = flat_urls[i]
                failed_info += '\n' + str(response)

            # TODO: add download stats
            if sub_idx == -1:
                reconstructed[orig_idx] = save_path
            else:
                reconstructed[orig_idx][sub_idx] = save_path

        return reconstructed, failed_info

    def process_batched(self, samples):
        if self.download_field not in samples or not samples[
                self.download_field]:
            return samples

        batch_nested_urls = samples[self.download_field]

        reconstructed, failed_info = self.download_nested_urls(
            batch_nested_urls, self.save_dir)

        samples[self.download_field] = reconstructed

        if len(failed_info):
            logger.error(f'Failed files:\n{failed_info}')

        return samples
