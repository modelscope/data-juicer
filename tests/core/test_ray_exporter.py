import copy
import os
import os.path as osp
import shutil
import unittest

from data_juicer.utils.unittest_utils import TEST_TAG
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.mm_utils import load_images_byte


class TestRayExporter(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = f'{cur_dir}/tmp/{self.__class__.__name__}/{self._testMethodName}'
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.data = [
            {'text': 'hello', Fields.stats: {'score': 1}, HashKeys.hash: 'a1'},
            {'text': 'world', Fields.stats: {'score': 2}, HashKeys.hash: 'b2'},
            {'text': 'test', Fields.stats: {'score': 3}, HashKeys.hash: 'c3'}
        ]
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up temporary outputs"""

        self.dataset = None
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _pop_raw_data_keys(self, keys):
        res = copy.deepcopy(self.data)
        for d_i in res:
            for k in keys:
                d_i.pop(k, None)

        return res

    @TEST_TAG('ray')
    def test_json_not_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.json')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListEqual(
            data_list, 
            self._pop_raw_data_keys([Fields.stats, HashKeys.hash]))

    @TEST_TAG('ray')
    def test_jsonl_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.jsonl')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self.data)

    @TEST_TAG('ray')
    def test_parquet_keep_stats(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.parquet')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_parquet(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self._pop_raw_data_keys([HashKeys.hash]))

    @TEST_TAG('ray')
    def test_lance_keep_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.lance')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_lance(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self._pop_raw_data_keys([Fields.stats]))

    @TEST_TAG('ray')
    def test_webdataset_multi_images(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'images': [img1_path, img2_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path])},
            {
                'json': {
                    'text': 'world',
                    'images': [img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img2_path, img3_path])},
            {
                'json': {
                    'text': 'test',
                    'images': [img1_path, img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path, img3_path])}
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path)
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))

        for i in range(len(data)):
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(
                res_list[i]['jpgs'],
                [Image.open(io.BytesIO(v)) for v in data[i]['jpgs']]
            )


if __name__ == '__main__':
    unittest.main()