import os
import unittest
from unittest.mock import patch
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from data_juicer.config import init_configs
from data_juicer.core.data.dataset_builder import (rewrite_cli_datapath, 
                                                   parse_cli_datapath,
                                                   DatasetBuilder)
from data_juicer.core.data.config_validator import ConfigValidationError
from data_juicer.utils.unittest_utils import (DataJuicerTestCaseBase, 
                                              SKIPPED_TESTS)


@SKIPPED_TESTS.register_module()
class DatasetBuilderTest(DataJuicerTestCaseBase):

    def setUp(self):
        """Setup basic configuration for tests"""
        self.base_cfg = Namespace()
        self.base_cfg.dataset_path = None
        self.executor_type = 'local'

        # Get the directory where this test file is located
        test_file_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(test_file_dir)


    def test_rewrite_cli_datapath_local_single_file(self):
        dataset_path = "./data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': [dataset_path], 'type': 'ondisk', 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_local_directory(self):
        dataset_path = "./data"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': [dataset_path], 'type': 'ondisk', 'weight': 1.0}], ans)

    def test_rewrite_cli_datapath_hf(self):
        dataset_path = "hf-internal-testing/librispeech_asr_dummy"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual([{'path': 'hf-internal-testing/librispeech_asr_dummy',
                           'split': 'train',
                           'type': 'huggingface'}],
                         ans)

    def test_rewrite_cli_datapath_local_wrong_files(self):
        dataset_path = "./missingDir"
        self.assertRaisesRegex(ValueError, "Unable to load the dataset",
                               rewrite_cli_datapath, dataset_path)

    def test_rewrite_cli_datapath_with_weights(self):
        dataset_path = "0.5 ./data/sample.json ./data/sample.txt"
        ans = rewrite_cli_datapath(dataset_path)
        self.assertEqual(
            [{'path': ['./data/sample.json'], 'type': 'ondisk', 'weight': 0.5},
             {'path': ['./data/sample.txt'], 'type': 'ondisk', 'weight': 1.0}],
            ans)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    def test_rewrite_cli_datapath_local_files(self, mock_isfile, mock_isdir):
        # Mock os.path.isdir and os.path.isfile to simulate local files
        mock_isfile.side_effect = lambda x: x.endswith('.jsonl')
        mock_isdir.side_effect = lambda x: x.endswith('_dir')

        dataset_path = "1.0 ds1.jsonl 2.0 ds2_dir 3.0 ds3.jsonl"
        expected = [
            {'type': 'ondisk', 'path': ['ds1.jsonl'], 'weight': 1.0},
            {'type': 'ondisk', 'path': ['ds2_dir'], 'weight': 2.0},
            {'type': 'ondisk', 'path': ['ds3.jsonl'], 'weight': 3.0}
        ]
        result = rewrite_cli_datapath(dataset_path)
        self.assertEqual(result, expected)

    def test_rewrite_cli_datapath_huggingface(self):
        dataset_path = "1.0 huggingface/dataset"
        expected = [
            {'type': 'huggingface', 'path': 'huggingface/dataset', 'split': 'train'}
        ]
        result = rewrite_cli_datapath(dataset_path)
        self.assertEqual(result, expected)

    def test_rewrite_cli_datapath_invalid(self):
        dataset_path = "1.0 ./invalid_path"
        with self.assertRaises(ValueError):
            rewrite_cli_datapath(dataset_path)

    def test_parse_cli_datapath(self):
        dataset_path = "1.0 ds1.jsonl 2.0 ds2_dir 3.0 ds3.jsonl"
        expected_paths = ['ds1.jsonl', 'ds2_dir', 'ds3.jsonl']
        expected_weights = [1.0, 2.0, 3.0]
        paths, weights = parse_cli_datapath(dataset_path)
        self.assertEqual(paths, expected_paths)
        self.assertEqual(weights, expected_weights)

    def test_parse_cli_datapath_default_weight(self):
        dataset_path = "ds1.jsonl ds2_dir 2.0 ds3.jsonl"
        expected_paths = ['ds1.jsonl', 'ds2_dir', 'ds3.jsonl']
        expected_weights = [1.0, 1.0, 2.0]
        paths, weights = parse_cli_datapath(dataset_path)
        self.assertEqual(paths, expected_paths)
        self.assertEqual(weights, expected_weights)


    def test_parse_cli_datapath_edge_cases(self):
        # Test various edge cases
        test_cases = [
            # Empty string
            ("", [], []),
            # Single path
            ("file.txt", ['file.txt'], [1.0]),
            # Multiple spaces between items
            ("file1.txt     file2.txt", ['file1.txt', 'file2.txt'], [1.0, 1.0]),
            # Tab characters
            ("file1.txt\tfile2.txt", ['file1.txt', 'file2.txt'], [1.0, 1.0]),
            # Paths with spaces in them (quoted)
            ('"my file.txt" 1.5 "other file.txt"',
            ['my file.txt', 'other file.txt'],
            [1.0, 1.5]),
        ]
        
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")

    def test_parse_cli_datapath_valid_weights(self):
        # Test various valid weight formats
        test_cases = [
            ("1.0 file.txt", ['file.txt'], [1.0]),
            ("1.5 file1.txt 2.0 file2.txt", 
            ['file1.txt', 'file2.txt'], 
            [1.5, 2.0]),
            ("0.5 file1.txt file2.txt 1.5 file3.txt",
            ['file1.txt', 'file2.txt', 'file3.txt'],
            [0.5, 1.0, 1.5]),
            # Test integer weights
            ("1 file.txt", ['file.txt'], [1.0]),
            ("2 file1.txt 3 file2.txt",
            ['file1.txt', 'file2.txt'],
            [2.0, 3.0]),
        ]
        
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")

    def test_parse_cli_datapath_special_characters(self):
        # Test paths with special characters
        test_cases = [
            # Paths with hyphens and underscores
            ("my-file_1.txt", ['my-file_1.txt'], [1.0]),
            # Paths with dots
            ("path/to/file.with.dots.txt", ['path/to/file.with.dots.txt'], [1.0]),
            # Paths with special characters
            ("file#1.txt", ['file#1.txt'], [1.0]),
            # Mixed case with weight
            ("1.0 Path/To/File.TXT", ['Path/To/File.TXT'], [1.0]),
            # Multiple paths with special characters
            ("2.0 file#1.txt 3.0 path/to/file-2.txt",
            ['file#1.txt', 'path/to/file-2.txt'],
            [2.0, 3.0]),
        ]
        
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")

    def test_parse_cli_datapath_multiple_datasets(self):
        # Test multiple datasets with various weight combinations
        test_cases = [
            # Multiple datasets with all weights specified
            ("0.5 data1.txt 1.5 data2.txt 2.0 data3.txt",
            ['data1.txt', 'data2.txt', 'data3.txt'],
            [0.5, 1.5, 2.0]),
            # Mix of weighted and unweighted datasets
            ("data1.txt 1.5 data2.txt data3.txt",
            ['data1.txt', 'data2.txt', 'data3.txt'],
            [1.0, 1.5, 1.0]),
            # Multiple datasets with same weight
            ("2.0 data1.txt 2.0 data2.txt 2.0 data3.txt",
            ['data1.txt', 'data2.txt', 'data3.txt'],
            [2.0, 2.0, 2.0]),
        ]
    
        for input_path, expected_paths, expected_weights in test_cases:
            paths, weights = parse_cli_datapath(input_path)
            self.assertEqual(paths, expected_paths, 
                            f"Failed paths for input: {input_path}")
            self.assertEqual(weights, expected_weights, 
                            f"Failed weights for input: {input_path}")
        
    def test_builder_single_dataset_config(self):
        """Test handling of single dataset configuration"""
        # Setup single dataset config
        self.base_cfg.dataset = {
            'type': 'ondisk',
            'path': 'test.jsonl'
        }
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        # Verify config was converted to list
        self.assertIsInstance(builder.load_strategies, list)
        self.assertEqual(len(builder.load_strategies), 1)
        
        # Verify config content preserved
        strategy = builder.load_strategies[0]
        self.assertEqual(strategy.ds_config['type'], 'ondisk')
        self.assertEqual(strategy.ds_config['path'], 'test.jsonl')

    def test_builder_multiple_dataset_config(self):
        """Test handling of multiple dataset configurations"""
        # Setup multiple dataset config
        self.base_cfg.dataset = [
            {
                'type': 'ondisk',
                'path': 'test1.jsonl'
            },
            {
                'type': 'ondisk',
                'path': 'test2.jsonl'
            }
        ]
        
        builder = DatasetBuilder(self.base_cfg, self.executor_type)
        
        # Verify list handling
        self.assertEqual(len(builder.load_strategies), 2)
        
        # Verify each config
        self.assertEqual(builder.load_strategies[0].ds_config['path'], 'test1.jsonl')
        self.assertEqual(builder.load_strategies[1].ds_config['path'], 'test2.jsonl')

    def test_builder_none_dataset_config(self):
        """Test handling when both dataset and dataset_path are None"""
        self.base_cfg.dataset = None
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('dataset_path or dataset', str(context.exception))

    def test_builder_mixed_dataset_types(self):
        """Test validation of mixed dataset types"""
        self.base_cfg.dataset = [
            {
                'type': 'ondisk',
                'path': 'test1.jsonl'
            },
            {
                'type': 'remote',
                'source': 'some_source'
            }
        ]
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('Mixture of diff types', str(context.exception))

    def test_builder_multiple_remote_datasets(self):
        """Test validation of multiple remote datasets"""
        self.base_cfg.dataset = [
            {
                'type': 'remote',
                'source': 'source1'
            },
            {
                'type': 'remote',
                'source': 'source2'
            }
        ]
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('Multiple remote datasets', str(context.exception))

    def test_builder_empty_dataset_config(self):
        """Test handling of empty dataset configuration"""
        self.base_cfg.dataset = []
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('dataset_path or dataset', str(context.exception))

    def test_builder_invalid_dataset_config_type(self):
        """Test handling of invalid dataset configuration type"""
        self.base_cfg.dataset = "invalid_string_config"
        
        with self.assertRaises(ConfigValidationError) as context:
            DatasetBuilder(self.base_cfg, self.executor_type)
        
        self.assertIn('Dataset config should be a dictionary', 
                      str(context.exception))

    def test_builder_ondisk_config(self):
        test_config_file = './data/test_config.yaml'
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_config_file}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'dataset-ondisk-json')
            self.assertEqual(cfg.dataset,
                             {'path': ['sample.json'], 'type': 'ondisk'})
            self.assertEqual(not cfg.dataset_path, True)

    def test_builder_ondisk_config_list(self):
        test_config_file = './data/test_config_list.yaml'
        out = StringIO()
        with redirect_stdout(out):
            cfg = init_configs(args=f'--config {test_config_file}'.split())
            self.assertIsInstance(cfg, Namespace)
            self.assertEqual(cfg.project_name, 'dataset-ondisk-list')
            self.assertEqual(cfg.dataset,[
                {'path': ['sample.json'], 'type': 'ondisk'},
                {'path': ['sample.txt'], 'type': 'ondisk'}])
            self.assertEqual(not cfg.dataset_path, True)

if __name__ == '__main__':
    unittest.main()
