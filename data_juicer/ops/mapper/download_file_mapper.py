import os
import os.path as osp

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import download_file, is_remote_path

from ..base_op import OPERATORS, Mapper

OP_NAME = 'download_file_mapper'


@OPERATORS.register_module(OP_NAME)
class DownloadFileMapper(Mapper):
    """Mapper to download url files to local files.
    """

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

    def _download_data_with_context(self, sample, context):
        """
        Download files with contexts.
        """
        raw_urls = sample[self.download_field]
        if isinstance(raw_urls, str):
            raw_urls = [raw_urls]

        new_paths = []
        response = None

        for raw_url in raw_urls:
            if is_remote_path(raw_url):
                save_path = osp.join(self.save_dir, osp.basename(raw_url))
                if not osp.exists(save_path):
                    response = download_file(raw_url,
                                             save_path,
                                             stream=self.stream,
                                             headers=self.headers,
                                             max_retries=self.max_retries,
                                             timeout=self.timeout,
                                             retry_delay=self.retry_delay,
                                             max_delay=self.max_delay)
                local_path = save_path
            else:
                local_path = raw_url

            if context and local_path not in sample[Fields.context]:
                if is_remote_path(raw_url) and response:
                    data_item = response.content
                else:
                    with open(local_path, 'rb') as f:
                        data_item = f.read()

                # store the data bytes into context
                sample[Fields.context][local_path] = data_item

            new_paths.append(local_path)

        # replace original url path with local path
        sample[self.download_field] = new_paths[0] if isinstance(
            sample[self.download_field], str) else new_paths
        return sample

    def process_single(self, sample, context=False):
        # there is no image in this sample
        if self.download_field not in sample or not sample[
                self.download_field]:
            return sample
        sample = self._download_data_with_context(sample, context)
        return sample
