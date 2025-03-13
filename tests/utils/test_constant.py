import os
import unittest

from data_juicer.core import NestedDataset
from data_juicer.config import init_configs
from data_juicer.utils.constant import StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

test_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              '..',
                              'config',
                              'demo_4_test.yaml')

class StatsKeysTest(DataJuicerTestCaseBase):

    def tearDown(cls) -> None:
        super().tearDown()
        StatsKeys._accessed_by = {}

    def test_basic_func(self):
        cfg = init_configs(args=f'--config {test_yaml_path}'.split())
        res = StatsKeys.get_access_log(cfg)
        self.assertEqual(len(res), 1)  # only 1 filter
        self.assertIn('language_id_score_filter', res)
        self.assertEqual(res['language_id_score_filter'], {'lang', 'lang_score'})

        # obtain again
        res_2 = StatsKeys.get_access_log(cfg)
        self.assertEqual(res, res_2)

    def test_basic_func_with_dataset(self):
        dataset = NestedDataset.from_list([{'text': 'hello world'}])
        cfg = init_configs(args=f'--config {test_yaml_path}'.split())
        res = StatsKeys.get_access_log(cfg, dataset)
        self.assertEqual(len(res), 1)  # only 1 filter
        self.assertIn('language_id_score_filter', res)
        self.assertEqual(res['language_id_score_filter'], {'lang', 'lang_score'})


if __name__ == '__main__':
    unittest.main()
