import os
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from jsonargparse import Namespace

from data_juicer.config import init_configs, get_default_cfg
from data_juicer.ops import load_ops
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

test_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'demo_4_test.yaml')

test_bad_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              'demo_4_test_bad_val.yaml')

WORKDIR = os.path.join(os.getcwd(), 'outputs/demo')

class ConfigTest(DataJuicerTestCaseBase):

    def test_help_info(self):
        out = StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            _ = init_configs(args=['--help'])
        out_str = out.getvalue()
        # self.assertIn('usage:', out_str, 'lacks message for command beginning')
        self.assertIn('--config CONFIG', out_str,
                      'lacks message for positional argument')
        self.assertIn('--project_name PROJECT_NAME', out_str,
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
                cfg.process[0], {
                    'whitespace_normalization_mapper': {
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                }, 'nested dict load fail, for nonparametric op')
            self.assertDictEqual(
                cfg.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                }, 'nested dict load fail, un-expected internal value')

            ops_from_cfg = load_ops(cfg.process)
            self.assertTrue(len(ops_from_cfg) == 4)

    def test_val_range_check_cmd(self):
        out = StringIO()
        err_msg_head = ("remove_table_text_mapper.min_col")
        err_msg = ("Input should be greater than or equal to 2")
        with redirect_stdout(out), redirect_stderr(out):
            with self.assertRaises(SystemExit) as cm:
                init_configs(
                    args=f'--config {test_yaml_path} '
                          '--remove_table_text_mapper.min_col 1'.split())
            self.assertEqual(cm.exception.code, 2)
        out_str = out.getvalue()
        self.assertIn(err_msg_head, out_str)
        self.assertIn(err_msg, out_str)

    def _test_val_range_check_yaml(self):
        out = StringIO()
        err_msg_head = ("remove_table_text_mapper.max_col")
        err_msg = ("Input should be less than or equal to 20")
        with redirect_stdout(out), redirect_stderr(out):
            with self.assertRaises(SystemExit) as cm:
                init_configs(args=f'--config {test_bad_yaml_path}'.split())
            self.assertEqual(cm.exception.code, 2)
        out_str = out.getvalue()
        self.assertIn(err_msg_head, out_str)
        self.assertIn(err_msg, out_str)

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
            print(f'ori_cfg.process[1] = {ori_cfg.process[1]}')
            self.assertDictEqual(
                ori_cfg.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_1.process[1], {
                    'language_id_score_filter': {
                        'lang': 'en',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_2.process[1], {
                    'language_id_score_filter': {
                        'lang': 'fr',
                        'min_score': 0.8,
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_3.process[1], {
                    'language_id_score_filter': {
                        'lang': 'zh',
                        'min_score': 0.6,
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })
            self.assertDictEqual(
                mixed_cfg_4.process[1], {
                    'language_id_score_filter': {
                        'lang': 'en',
                        'min_score': 0.5,
                        'text_key': 'text',
                        'image_key': 'images',
                        'audio_key': 'audios',
                        'video_key': 'videos',
                        'query_key': 'query',
                        'response_key': 'response',
                        'history_key': 'history',
                        'accelerator': None,
                        'num_proc': 4,
                        'stats_export_path': None,
                        'cpu_required': 1,
                        'mem_required': 0,
                        'turbo': False,
                        'batch_size': 1000,
                        'index_key': None,
                        'skip_op_error': True,
                        'work_dir': WORKDIR,
                    }
                })

    def test_op_params_parsing(self):
        from jsonargparse import ArgumentParser
        from data_juicer.config.config import (sort_op_by_types_and_names, _collect_config_info_from_class_docs)
        from data_juicer.ops.base_op import OPERATORS

        base_class_params = {
            'text_key', 'image_key', 'audio_key', 'video_key', 'query_key', 'response_key', 'history_key',
            'accelerator', 'turbo', 'batch_size', 'num_proc', 'cpu_required', 'mem_required', 'work_dir',
        }

        parser = ArgumentParser(default_env=True, default_config_files=None)
        ops_sorted_by_types = sort_op_by_types_and_names(
            OPERATORS.modules.items())
        op_params = _collect_config_info_from_class_docs(ops_sorted_by_types,
                                                         parser)

        for op_name, params in op_params.items():
            for base_param in base_class_params:
                base_param_key = f'{op_name}.{base_param}'
                self.assertIn(base_param_key, params)


    def test_get_default_cfg(self):
        """Test getting default configuration from config_all.yaml"""
        # Get default config
        cfg = get_default_cfg()
        
        # Verify basic default values
        self.assertIsInstance(cfg, Namespace)
        
        # Test essential defaults
        self.assertEqual(cfg.executor_type, 'default')
        self.assertEqual(cfg.ray_address, 'auto')
        self.assertEqual(cfg.text_keys, 'text')
        self.assertEqual(cfg.add_suffix, False)
        self.assertEqual(cfg.export_path, './outputs/')
        self.assertEqual(cfg.suffixes, None)
        
        # Test default values are of correct type
        self.assertIsInstance(cfg.executor_type, str)
        self.assertIsInstance(cfg.add_suffix, bool)
        self.assertIsInstance(cfg.export_path, str)

if __name__ == '__main__':
    unittest.main()
