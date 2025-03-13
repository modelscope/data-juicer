import unittest

from data_juicer.utils.model_utils import get_backup_model_link
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# other funcs are called by ops already
class ModelUtilsTest(DataJuicerTestCaseBase):

    def test_get_backup_model_link(self):
        test_data = [
            ('lid.176.bin', 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/'),  # exact match
            ('zh.sp.model', 'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/'),  # pattern match
            ('invalid_model_name', None),  # invalid model name
        ]
        for model_name, expected_link in test_data:
            self.assertEqual(get_backup_model_link(model_name), expected_link)


if __name__ == '__main__':
    unittest.main()
