import asyncio
import copy
import os
import os.path as osp
from typing import List, Union
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from data_juicer.utils.file_utils import download_file, is_remote_path

from ..base_op import OPERATORS, Mapper

OP_NAME = "download_file_mapper"


@OPERATORS.register_module(OP_NAME)
class DownloadFileMapper(Mapper):
    """Mapper to download url files to local files."""

    _batched_op = True

    def __init__(
        self,
        save_dir: str = None,
        download_field: str = None,
        timeout: int = 30,
        max_concurrent: int = 10,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param save_dir: The directory to save downloaded files.
        :param download_field: The filed name to get the url to download.
        :param max_concurrent: Maximum concurrent downloads.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.download_field = download_field
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    def download_files_async(self, urls, save_dir, **kwargs):

        async def _download_file(
            session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, idx: int, url: str, save_dir, **kwargs
        ) -> dict:
            try:
                filename = os.path.basename(urlparse(url).path)
                save_path = osp.join(save_dir, filename)
                status = "success"
                if os.path.exists(save_path):
                    return idx, save_path, status, None

                async with semaphore:
                    response = await download_file(session, url, save_path, timeout=self.timeout, **kwargs)
            except Exception as e:
                status = "failed"
                response = str(e)
                save_path = None

            return idx, save_path, status, response

        async def run_downloads(urls, save_dir, **kwargs):
            semaphore = asyncio.Semaphore(self.max_concurrent)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    _download_file(session, semaphore, idx, url, save_dir, **kwargs) for idx, url in enumerate(urls)
                ]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_downloads(urls, save_dir, **kwargs))
        results.sort(key=lambda x: x[0])

        return results

    def download_nested_urls(self, nested_urls: List[Union[str, List[str]]], save_dir: str):
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
                    structure_info.append((idx, -1))  # -1 means single str element

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

        failed_info = ""
        for i, (idx, save_path, status, response) in enumerate(download_results):
            orig_idx, sub_idx = structure_info[i]
            if status != "success":
                save_path = flat_urls[i]
                failed_info += "\n" + str(response)

            # TODO: add download stats
            if sub_idx == -1:
                reconstructed[orig_idx] = save_path
            else:
                reconstructed[orig_idx][sub_idx] = save_path

        return reconstructed, failed_info

    def process_batched(self, samples):
        if self.download_field not in samples or not samples[self.download_field]:
            return samples

        batch_nested_urls = samples[self.download_field]

        reconstructed, failed_info = self.download_nested_urls(batch_nested_urls, self.save_dir)

        samples[self.download_field] = reconstructed

        if len(failed_info):
            logger.error(f"Failed files:\n{failed_info}")

        return samples
