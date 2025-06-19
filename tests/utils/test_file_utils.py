import os
import regex as re
import requests
import tempfile
import shutil
import jsonlines
import unittest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError

import data_juicer
from data_juicer.utils.file_utils import (
    find_files_with_suffix, is_absolute_path, download_file,
    add_suffix_to_filename, create_directory_if_not_exists, transfer_filename,
    copy_data
)
from data_juicer.utils.mm_utils import Fields
from data_juicer.utils.logger_utils import setup_logger
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FileUtilsTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        self.temp_output_path = 'tmp/test_file_utils/'
        os.makedirs(self.temp_output_path)

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')

    def test_find_files_with_suffix(self):
        # prepare test files
        fn_list = ['test1.txt', 'test2.txt', 'test3.md']
        for fn in fn_list:
            with open(os.path.join(self.temp_output_path, fn), 'w') as f:
                f.write(fn)

        self.assertEqual(find_files_with_suffix(os.path.join(self.temp_output_path, 'test1.txt')),
                         {'.txt': [os.path.join(self.temp_output_path, 'test1.txt')]})
        self.assertEqual(find_files_with_suffix(self.temp_output_path),
                         {'.txt': [os.path.join(self.temp_output_path, 'test1.txt'), os.path.join(self.temp_output_path, 'test2.txt')],
                          '.md': [os.path.join(self.temp_output_path, 'test3.md')]})
        self.assertEqual(find_files_with_suffix(self.temp_output_path, 'txt'),
                         {'.txt': [os.path.join(self.temp_output_path, 'test1.txt'), os.path.join(self.temp_output_path, 'test2.txt')]})

    def test_is_absolute_path(self):
        self.assertFalse(is_absolute_path(self.temp_output_path))
        self.assertTrue(is_absolute_path(os.path.abspath(self.temp_output_path)))

    def test_add_suffix_to_filename(self):
        self.assertEqual(add_suffix_to_filename('test.txt', '_suffix'), 'test_suffix.txt')
        self.assertEqual(add_suffix_to_filename('test.txt', ''), 'test.txt')
        self.assertEqual(add_suffix_to_filename('test', '_suffix'), 'test_suffix')
        self.assertEqual(add_suffix_to_filename('.git', '_suffix'), '.git_suffix')

    def test_create_directory_if_not_exists(self):
        self.assertTrue(os.path.exists(self.temp_output_path))
        create_directory_if_not_exists(self.temp_output_path)
        self.assertTrue(os.path.exists(self.temp_output_path))
        os.rmdir(self.temp_output_path)
        self.assertFalse(os.path.exists(self.temp_output_path))
        create_directory_if_not_exists(self.temp_output_path)
        self.assertTrue(os.path.exists(self.temp_output_path))

    def test_transfer_filename(self):
        self.assertTrue(
            re.match(
                os.path.join(self.temp_output_path, Fields.multimodal_data_output_dir, 'op1', 'abc__dj_hash_#(.*?)#.jpg'),
                transfer_filename(os.path.join(self.temp_output_path, 'abc.jpg'), 'op1')))

    def test_copy_data(self):
        tgt_fn = 'test.txt'
        ori_dir = os.path.join(self.temp_output_path, 'test1')
        tgt_dir = os.path.join(self.temp_output_path, 'test2')

        self.assertFalse(copy_data(ori_dir, tgt_dir, tgt_fn))

        os.makedirs(ori_dir, exist_ok=True)
        with open(os.path.join(ori_dir, tgt_fn), 'w') as f:
            f.write('test')

        self.assertTrue(copy_data(ori_dir, tgt_dir, tgt_fn))
        self.assertTrue(os.path.exists(os.path.join(tgt_dir, tgt_fn)))


class TestDownloadFile(DataJuicerTestCaseBase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.test_dir, "test_file.txt")
        self.url = "http://example.com/file"
        self.headers = {"User-Agent": "test"}
        data_juicer.utils.logger_utils.LOGGER_SETUP = False

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def get_warning_logs(self):
        file_path = os.path.join(self.test_dir, 'log_WARNING.txt')
        with jsonlines.open(file_path, 'r') as reader:
            records = [line for line in reader]
        return records

    @patch("data_juicer.utils.file_utils.requests.get")
    def test_500_error_with_retry_success(self, mock_get):
        setup_logger(self.test_dir)

        # first request 500 second 200
        mock_resp1 = MagicMock(status_code=500)
        mock_resp2 = MagicMock(status_code=200)
        # mock_resp2.iter_content.return_value = [b"data"]  # stream
        mock_resp2.content = b"data"
        mock_get.side_effect = [mock_resp1, mock_resp2]

        response = download_file(
            url=self.url,
            save_path=self.test_path,
            headers=self.headers,
            max_retries=3,
            retry_delay=0.1
        )
        
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(response, mock_resp2)

        records = self.get_warning_logs()
        self.assertEqual(len(records), 2)

        self.assertIn("500 Server Error", records[0]['text'])
        self.assertIn('Will retry in', records[1]['text'])
        self.assertIn('(attempt 1)', records[1]['text'])
        with open(self.test_path, 'r') as f:
            self.assertEqual(f.read(), 'data')

    @patch("data_juicer.utils.file_utils.requests.get")
    def test_500_error_exceed_max_retries(self, mock_get):
        setup_logger(self.test_dir)

        # mock 500 errors for four times (max_retries=3)
        mock_resp = MagicMock(status_code=500)
        
        mock_resp.raise_for_status.side_effect = HTTPError("500 Server Error", response=mock_resp)
        mock_get.return_value = mock_resp

        with self.assertRaises(HTTPError) as cm:
            download_file(
                url=self.url,
                save_path=self.test_path,
                max_retries=3,
                retry_delay=0.1
            )
        
        self.assertEqual(mock_get.call_count, 4)  # initial request + 3 retries

        records = self.get_warning_logs()
        self.assertEqual(len(records), 8)

        self.assertIn("500 Server Error", records[0]['text'])

        self.assertIn('Will retry in', records[1]['text'])
        self.assertIn('(attempt 1)', records[1]['text'])
        
        self.assertIn('Will retry in', records[5]['text'])
        self.assertIn('(attempt 3)', records[5]['text'])

        self.assertIn("500 Server Error", records[6]['text'])
        self.assertIn("Reach the maximum retry times", records[7]['text'])

    @patch("data_juicer.utils.file_utils.requests.get")
    def test_400_client_error(self, mock_get):
        setup_logger(self.test_dir)

        mock_resp = MagicMock(status_code=404)
        mock_resp.raise_for_status.side_effect = HTTPError("404 Client Error", response=mock_resp)
        mock_get.return_value = mock_resp

        with self.assertRaises(HTTPError) as cm:
            download_file(
                url=self.url,
                save_path=self.test_path
            )
        
        self.assertIn("404 Client Error", str(cm.exception))

    @patch("data_juicer.utils.file_utils.requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        with self.assertRaises(requests.ConnectionError):
            download_file(
                url=self.url,
                save_path=self.test_path,
                max_retries=2
            )
            
        self.assertEqual(mock_get.call_count, 1)

    @patch("data_juicer.utils.file_utils.requests.get")
    def test_timeout_error(self, mock_get):
        mock_get.side_effect = requests.Timeout("Request timed out")

        with self.assertRaises(requests.Timeout):
            download_file(
                url=self.url,
                save_path=self.test_path,
                timeout=1,
                max_retries=1
            )
        
        self.assertEqual(mock_get.call_count, 1)


if __name__ == '__main__':
    unittest.main()
