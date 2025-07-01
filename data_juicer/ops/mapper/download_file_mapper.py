import asyncio
import copy
import os
import os.path as osp
from typing import List, Union
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import download_file, is_remote_path

from ..base_op import OPERATORS, Mapper

OP_NAME = "download_file_mapper"


@OPERATORS.register_module(OP_NAME)
class DownloadFileMapper(Mapper):
    """Mapper to download url files to local files or load them into memory."""

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
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.timeout = timeout
        self.max_concurrent = max_concurrent

    def download_files_async(self, urls, save_dir=None, context=False, **kwargs):

        async def _download_file(
            session: aiohttp.ClientSession,
            semaphore: asyncio.Semaphore,
            idx: int,
            url: str,
            save_dir=None,
            context=False,
            **kwargs,
        ) -> dict:
            try:
                status, response, content, save_path = "success", None, None, None

                if not is_remote_path(url):
                    if context:
                        with open(url, "rb") as f:
                            content = f.read()
                    if save_dir:
                        save_path = url
                    return idx, save_path, status, response, content

                if save_dir:
                    filename = os.path.basename(urlparse(url).path)
                    save_path = osp.join(save_dir, filename)
                    if os.path.exists(save_path):
                        if context:
                            with open(save_path, "rb") as f:
                                content = f.read()
                        return idx, save_path, status, response, content

                async with semaphore:
                    response, content = await download_file(
                        session, url, save_path, return_content=True, timeout=self.timeout, **kwargs
                    )
            except Exception as e:
                status = "failed"
                response = str(e)
                save_path = None
                content = None

            return idx, save_path, status, response, content

        async def run_downloads(urls, save_dir=None, context=False, **kwargs):
            semaphore = asyncio.Semaphore(self.max_concurrent)
            async with aiohttp.ClientSession() as session:
                tasks = [
                    _download_file(session, semaphore, idx, url, save_dir, context, **kwargs)
                    for idx, url in enumerate(urls)
                ]
                return await asyncio.gather(*tasks)

        results = asyncio.run(run_downloads(urls, save_dir, context, **kwargs))
        results.sort(key=lambda x: x[0])

        return results

    def _flat_urls(self, nested_urls):
        flat_urls = []
        structure_info = []  # save as original index, sub index

        for idx, urls in enumerate(nested_urls):
            if isinstance(urls, list):
                for sub_idx, url in enumerate(urls):
                    flat_urls.append(url)
                    structure_info.append((idx, sub_idx))
            else:
                flat_urls.append(urls)
                structure_info.append((idx, -1))  # -1 means single str element

        return flat_urls, structure_info

    def _create_path_structrue(self, nested_urls, keep_failed_url=True) -> str:
        if keep_failed_url:
            reconstructed = copy.deepcopy(nested_urls)
        else:
            reconstructed = []
            for item in nested_urls:
                if isinstance(item, list):
                    reconstructed.append([None] * len(item))
                else:
                    reconstructed.append(None)

        return reconstructed

    def download_nested_urls(self, nested_urls: List[Union[str, List[str]]], save_dir=None, context_content=None):
        flat_urls, structure_info = self._flat_urls(nested_urls)

        download_results = self.download_files_async(
            flat_urls,
            save_dir,
            context=True if context_content is not None else False,
        )

        if self.save_dir:
            reconstructed_path = self._create_path_structrue(nested_urls)
        else:
            reconstructed_path = None

        failed_info = ""
        for i, (idx, save_path, status, response, content) in enumerate(download_results):
            orig_idx, sub_idx = structure_info[i]
            if status != "success":
                save_path = flat_urls[i]
                failed_info += "\n" + str(response)

            if context_content is not None:
                path_key = save_path if save_path else flat_urls[i]
                context_content[orig_idx].update({path_key: content})

            if self.save_dir:
                # TODO: add download stats
                if sub_idx == -1:
                    reconstructed_path[orig_idx] = save_path
                else:
                    reconstructed_path[orig_idx][sub_idx] = save_path

        return context_content, reconstructed_path, failed_info

    def process_batched(self, samples, context=False):
        if not self.save_dir:
            assert context, "context must be True when save_dir is not specified"

        if self.download_field not in samples or not samples[self.download_field]:
            return samples

        if context:
            if Fields.context in samples:
                context_content = samples[Fields.context]
            else:
                context_content = [{} for _ in range(len(samples[self.download_field]))]
        else:
            context_content = None

        batch_nested_urls = samples[self.download_field]

        context_content, reconstructed_path, failed_info = self.download_nested_urls(
            batch_nested_urls, save_dir=self.save_dir, context_content=context_content
        )

        if self.save_dir:
            samples[self.download_field] = reconstructed_path

        if context:
            samples[Fields.context] = context_content

        if len(failed_info):
            logger.error(f"Failed files:\n{failed_info}")

        return samples
