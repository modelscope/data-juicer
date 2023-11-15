import os
import unittest

from datasets import Dataset

from data_juicer.ops.deduplicator.image_deduplicator import \
    ImageDeduplicator


class ImageDeduplicatorTest(unittest.TestCase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', 'data')
    img1_path = os.path.join(data_path, 'img1.png')
    img2_path = os.path.join(data_path, 'img2.jpg')
    img3_path = os.path.join(data_path, 'img3.jpg')
    # img4.png is a duplicate sample of img1.png
    img4_path = os.path.join(data_path, 'img4.png')
    # img5.jpg is a duplicate sample of img2.jpg
    img5_path = os.path.join(data_path, 'img5.jpg')
    # img6.jpg is a duplicate sample of img3.jpg
    img6_path = os.path.join(data_path, 'img6.jpg')
    # img7.jpg is a duplicate sample of img6.jpg
    img7_path = os.path.join(data_path, 'img7.jpg')
 

    def _run_image_deduplicator(self,
                                dataset: Dataset, target_list,
                                op):

        dataset = dataset.map(op.compute_hash)
        dataset, _ = op.process(dataset)
        dataset = dataset.select_columns(column_names=[op.image_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_1(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_2(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img2_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_3(self):

        ds_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }, {
            'images': [self.img4_path]
        }, {
            'images': [self.img5_path]
        }, {
            'images': [self.img6_path]
        }, {
            'images': [self.img7_path]
        }]
        tgt_list = [{
            'images': [self.img1_path]
        }, {
            'images': [self.img2_path]
        }, {
            'images': [self.img3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_4(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img4_path, self.img5_path, self.img6_path]
        }, {
            'images': [self.img7_path]
        }, {
            'images': [self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path, self.img3_path]
        }, {
            'images': [self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_5(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator()
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_6(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='dhash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_7(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='whash')
        self._run_image_deduplicator(dataset, tgt_list, op)

    def test_8(self):

        ds_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img4_path, self.img5_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }, {
            'images': [self.img6_path, self.img6_path]
        }]
        tgt_list = [{
            'images': [self.img1_path, self.img2_path]
        }, {
            'images': [self.img2_path, self.img1_path]
        }, {
            'images': [self.img7_path, self.img7_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = ImageDeduplicator(method='ahash')
        self._run_image_deduplicator(dataset, tgt_list, op)

if __name__ == '__main__':
    unittest.main()
