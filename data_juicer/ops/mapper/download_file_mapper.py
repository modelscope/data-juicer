import copy
import os
from typing import List, Union

from loguru import logger

from data_juicer.utils.file_utils import download_file, is_remote_path

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
                 timeout: int = 30,
                 stream: bool = False,
                 chunk_size: int = 65536,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param save_dir: The directory to save downloaded files.
        :param download_field: The filed name to get the url to download.
        :param timeout: The timeout in seconds for each HTTP request.
        :param stream: If True, the file will be downloaded in chunks.
            If False, the entire file will be downloaded at once.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.download_field = download_field
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.timeout = timeout
        self.stream = stream
        self.chunk_size = chunk_size

    def download_files_async(self, urls, save_dir):
        import asyncio

        import aiohttp

        async def _download_file(session: aiohttp.ClientSession, idx: int,
                                 url: str, save_dir) -> dict:
            result_dict = await download_file(session,
                                              url,
                                              save_dir,
                                              timeout=self.timeout,
                                              stream=self.stream,
                                              chunk_size=self.chunk_size)
            result_dict.update({'idx': idx})
            return result_dict

        async def run_downloads():
            async with aiohttp.ClientSession() as session:
                tasks = [
                    _download_file(session, idx, url, save_dir)
                    for idx, url in enumerate(urls)
                ]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_downloads())
        results.sort(key=lambda x: x['idx'])

        return results

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

        download_results = self.download_files_async(
            flat_urls,
            save_dir,
        )

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
        for i, result_item in enumerate(download_results):
            orig_idx, sub_idx = structure_info[i]
            status = result_item['status']
            message = result_item['message']

            if status != 'success':
                save_path = flat_urls[i]
                failed_info += '\n' + str(message)
            else:
                save_path = result_item['save_path']

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
