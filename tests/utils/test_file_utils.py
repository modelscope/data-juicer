import os
import unittest
import regex as re

from data_juicer.utils.file_utils import (
    find_files_with_suffix, is_absolute_path,
    add_suffix_to_filename, create_directory_if_not_exists, transfer_filename,
    copy_data
)
from data_juicer.utils.mm_utils import Fields

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


if __name__ == '__main__':
    unittest.main()
