import unittest
import os
import shutil
import tempfile
import threading
import numpy as np
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.ops.mapper.download_file_mapper import DownloadFileMapper


class DownloadFileMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        self.img1_path = os.path.join(self.data_path, 'img1.png')
        self.img2_path = os.path.join(self.data_path, 'img2.jpg')
        self.img3_path = os.path.join(self.data_path, 'img3.jpg')

        # start HTTP server
        self.server_address = ('localhost', 0)  # 0 means random port
        self.httpd = HTTPServer(
            self.server_address,
            partial(SimpleHTTPRequestHandler, directory=self.data_path)
        )
        self.port = self.httpd.server_address[1]
        self.img1_url = f'http://localhost:{self.port}/{os.path.basename(self.img1_path)}'
        self.img2_url = f'http://localhost:{self.port}/{os.path.basename(self.img2_path)}'
        self.img3_url = f'http://localhost:{self.port}/{os.path.basename(self.img3_path)}'
        
        # start the server in a thread
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.temp_dir)

    def _test_image_download(self, ds_list):
        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images')

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            for s_path, r_path in zip(source[op.image_key], res[op.image_key]):
                fname = os.path.basename(s_path)
                self.assertEqual(fname, os.path.basename(r_path))
                if s_path.startswith('http'):
                    self.assertEqual(os.path.dirname(r_path), self.temp_dir)
                else:
                    self.assertEqual(s_path, r_path)

                t_img = np.array(load_image(os.path.join(self.data_path, fname)))
                r_img = np.array(load_image(r_path))

                np.testing.assert_array_equal(t_img, r_img)

    def test_image_download(self):
        ds_list = [{
            'images': [self.img1_url],
            'id': 1
        }, {
            'images': [self.img2_url, self.img3_url],
            'id': 2
        }, {
            'images': [self.img1_url, self.img2_url, self.img3_url],
            'id': 3
        }]

        self._test_image_download(ds_list)
        
    def test_image_url_and_local_path(self):
        ds_list = [{
            'images': [self.img1_path],
            'id': 1
        }, {
            'images': [self.img2_path, self.img3_url],
            'id': 2
        }, {
            'images': [self.img1_path, self.img2_url, self.img3_path],
            'id': 3
        }]

        self._test_image_download(ds_list)
        
    def test_download_image_failed(self):
        ds_list = [{
            'images': self.img2_url + '_failed_test',
            'id': 1
        }, {
            'images': self.img3_url + '_failed_test',
            'id': 2
        }, {
            'images': self.img1_url,
            'id': 3
        }]

        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images')

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            s_path, r_path = source[op.image_key], res[op.image_key]
            fname = os.path.basename(s_path)
            self.assertEqual(fname, os.path.basename(r_path))
            if s_path.startswith('http') and 'failed_test' not in s_path:
                self.assertEqual(os.path.dirname(r_path), self.temp_dir)
            else:
                self.assertEqual(s_path, r_path)

    def test_image_str_type(self):
        ds_list = [{
            'images': self.img2_path,
            'id': 1
        }, {
            'images': self.img3_path,
            'id': 2
        }, {
            'images': self.img1_url,
            'id': 3
        }]

        op = DownloadFileMapper(
                save_dir=self.temp_dir,
                download_field='images')

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        
        res_list = dataset.to_list()
        res_list = sorted(res_list, key=lambda x: x['id'])

        self.assertEqual(len(ds_list), len(res_list))

        for i in range(len(ds_list)):
            source, res = ds_list[i], res_list[i]
            s_path, r_path = source[op.image_key], res[op.image_key]
            fname = os.path.basename(s_path)
            self.assertEqual(fname, os.path.basename(r_path))
            if s_path.startswith('http'):
                self.assertEqual(os.path.dirname(r_path), self.temp_dir)
            else:
                self.assertEqual(s_path, r_path)

            t_img = np.array(load_image(os.path.join(self.data_path, fname)))
            r_img = np.array(load_image(r_path))

            np.testing.assert_array_equal(t_img, r_img)


if __name__ == '__main__':
    unittest.main()
