import unittest

from copy import deepcopy

from data_juicer.ops.mapper.simple_aug_en_mapper import SimpleAugEnMapper


class SimpleAugEnMapperTest(unittest.TestCase):

    def setUp(self):
        self.samples = {
            'text': ['I am a deep learning engineer. I love LLM.'],
            'meta': ['meta information'],
        }

    def test_number_of_generated_samples_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 3
        op = SimpleAugEnMapper(
            sequential=True,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_char=True,
            spelling_error_word=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = op.process(self.samples)
        self.assertEqual(len(result['text']), aug_num + 1)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_number_of_generated_samples_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 3
        op = SimpleAugEnMapper(
            sequential=False,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_char=True,
            spelling_error_word=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = op.process(self.samples)
        self.assertEqual(len(result['text']), aug_num * aug_method_num + 1)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 0
        # sequential on
        op = SimpleAugEnMapper(
            sequential=True,
            aug_num=aug_num,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = op.process(self.samples)
        self.assertEqual(len(result['text']), 1)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_zero_aug_methods_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 0
        # sequential off
        op = SimpleAugEnMapper(
            sequential=False,
            aug_num=aug_num,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = op.process(self.samples)
        self.assertEqual(len(result['text']), 1)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_on(self):
        aug_num = 3
        aug_method_num = 9
        # sequential on
        op = SimpleAugEnMapper(
            sequential=True,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_word=True,
            spelling_error_word=True,
            split_random_word=True,
            keyboard_error_char=True,
            ocr_error_char=True,
            delete_random_char=True,
            swap_random_char=True,
            insert_random_char=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = op.process(self.samples)
        self.assertEqual(len(result['text']), aug_num + 1)
        self.assertEqual(len(result['meta']), len(result['text']))

    def test_all_aug_methods_with_sequential_off(self):
        aug_num = 3
        aug_method_num = 9
        # sequential off
        op = SimpleAugEnMapper(
            sequential=False,
            aug_num=aug_num,
            delete_random_word=True,
            swap_random_word=True,
            spelling_error_word=True,
            split_random_word=True,
            keyboard_error_char=True,
            ocr_error_char=True,
            delete_random_char=True,
            swap_random_char=True,
            insert_random_char=True,
        )
        self.assertEqual(len(op.aug), aug_method_num)
        result = op.process(self.samples)
        self.assertEqual(len(result['text']), aug_num * aug_method_num + 1)
        self.assertEqual(len(result['meta']), len(result['text']))


if __name__ == '__main__':
    unittest.main()
