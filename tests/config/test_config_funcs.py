import os
import unittest
from contextlib import redirect_stdout
from io import StringIO

from jsonargparse import Namespace

from data_juicer.config import init_configs
from data_juicer.ops import load_ops

test_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'demo_4_test.yaml')


class ConfigTest(unittest.TestCase):

    def test_help_info(self):
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            _ = init_configs(args=['--help'])
        out_str = out.getvalue()
        self.assertIn('usage:', out_str, 'lacks message for command beginning')
        self.assertIn('--config CONFIG', out_str,
                      'lacks message for positional argument')
        self.assertIn('[--project_name PROJECT_NAME]', out_str,
                      'lacks message for optional argument')
        self.assertIn(
            'Number of processes to process dataset. (type:', out_str,
            'the help message of `np` argument does not show as expected')

    def test_yaml_cfg_file(self):
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_yaml_path}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'test_demo')
            self.assertDictEqual(
                cfg.process[0],
                {'whitespace_normalization_mapper': {
                    'text_key': 'text'
                }}, 'nested dict load fail, for nonparametric op')
            self.assertDictEqual(
                cfg.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.8,
                        'text_key': 'text'
                    }
                }, 'nested dict load fail, un-expected internal value')

            _, op_from_cfg = load_ops(cfg.process)
            self.assertTrue(len(op_from_cfg) == 3)

    def test_mixture_cfg(self):
        out = StringIO()
        with redirect_stdout(out):
            ori_cfg = init_configs(args=f'--config {test_yaml_path}'.split())
            mixed_cfg_1 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang en'.split())
            mixed_cfg_2 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang=fr'.split())
            mixed_cfg_3 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang zh '
                '--language_id_score_filter.min_score 0.6'.split())
            mixed_cfg_4 = init_configs(
                args=f'--config {test_yaml_path} '
                '--language_id_score_filter.lang=en '
                '--language_id_score_filter.min_score=0.5'.split())
            self.assertDictEqual(
                ori_cfg.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.8,
                        'text_key': 'text'
                    }
                })
            self.assertDictEqual(
                mixed_cfg_1.process[1], {
                    'language_id_score_filter': {
                        'lang': 'en',
                        'min_score': 0.8,
                        'text_key': 'text'
                    }
                })
            self.assertDictEqual(
                mixed_cfg_2.process[1], {
                    'language_id_score_filter': {
                        'lang': 'fr',
                        'min_score': 0.8,
                        'text_key': 'text'
                    }
                })
            self.assertDictEqual(
                mixed_cfg_3.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.6,
                        'text_key': 'text'
                    }
                })
            self.assertDictEqual(
                mixed_cfg_4.process[1], {
                    'language_id_score_filter': {
                        'lang': 'en',
                        'min_score': 0.5,
                        'text_key': 'text'
                    }
                })


if __name__ == '__main__':
    unittest.main()
