import unittest

from data_juicer.ops.load import load_ops

class OpFusionTest(unittest.TestCase):

    def _run_op_fusion(self, original_process_list, target_process_list):
        new_process_list, _ = load_ops(original_process_list, op_fusion=True)
        self.assertEqual(new_process_list, target_process_list)

    def test_regular_config(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)': [{  # noqa: E501
                'words_num_filter': {
                    'lang': 'en',
                    'max_num': 100000,
                    'min_num': 20,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'word_repetition_filter': {
                    'lang': 'en',
                    'max_ratio': 0.19,
                    'min_ratio': 0.0,
                    'rep_len': 5,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'stopwords_filter': {
                    'lang': 'en',
                    'min_ratio': 0.3,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'flagged_words_filter': {
                    'lang': 'en',
                    'max_ratio': 0.01,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'perplexity_filter': {
                    'lang': 'en',
                    'max_ppl': 1500,
                    'text_key': 'text'
                }
            }]
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_only_mapper(self):
        original_process = [{
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_only_deduplicator(self):

        original_process = [{
            'document_deduplicator': {
                'ignore_non_character': True,
                'lowercase': True,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }, {
            'document_minhash_deduplicator': {
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6,
                'num_permutations': 256,
                'jaccard_threshold': 0.7
            }
        }]
        target_process = [{
            'document_deduplicator': {
                'ignore_non_character': True,
                'lowercase': True,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }, {
            'document_minhash_deduplicator': {
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6,
                'num_permutations': 256,
                'jaccard_threshold': 0.7
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_non_fusible_filters(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'alphanumeric_filter': {
                'min_ratio': 0.25,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'alphanumeric_filter': {
                'min_ratio': 0.25,
                'text_key': 'text'
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_not_enough_fusible_ops_to_fuse(self):
        # still apply reordering:
        # - ordinary ops
        # - ops with InterVars.lines
        # - ops with InterVars.words
        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_multiple_groups(self):

        original_process = [{
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'OpFusion:(stopwords_filter,flagged_words_filter)': [{
                'stopwords_filter': {
                    'lang': 'en',
                    'min_ratio': 0.3,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'flagged_words_filter': {
                    'lang': 'en',
                    'max_ratio': 0.01,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }]
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'OpFusion:(words_num_filter,word_repetition_filter,perplexity_filter)': [{  # noqa: E501
                'words_num_filter': {
                    'lang': 'en',
                    'max_num': 100000,
                    'min_num': 20,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'word_repetition_filter': {
                    'lang': 'en',
                    'max_ratio': 0.19,
                    'min_ratio': 0.0,
                    'rep_len': 5,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'perplexity_filter': {
                    'lang': 'en',
                    'max_ppl': 1500,
                    'text_key': 'text'
                }
            }]
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        self._run_op_fusion(original_process, target_process)

    def test_only_fusible_ops(self):

        original_process = [{
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }]
        target_process = [{
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)': [{  # noqa: E501
                'words_num_filter': {
                    'lang': 'en',
                    'max_num': 100000,
                    'min_num': 20,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'word_repetition_filter': {
                    'lang': 'en',
                    'max_ratio': 0.19,
                    'min_ratio': 0.0,
                    'rep_len': 5,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'stopwords_filter': {
                    'lang': 'en',
                    'min_ratio': 0.3,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'flagged_words_filter': {
                    'lang': 'en',
                    'max_ratio': 0.01,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'perplexity_filter': {
                    'lang': 'en',
                    'max_ppl': 1500,
                    'text_key': 'text'
                }
            }]
        }]
        self._run_op_fusion(original_process, target_process)

    def test_different_intermediate_vars(self):

        original_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'words_num_filter': {
                'lang': 'en',
                'max_num': 100000,
                'min_num': 20,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'average_line_length_filter': {
                'min_len': 10,
                'text_key': 'text'
            }
        }, {
            'word_repetition_filter': {
                'lang': 'en',
                'max_ratio': 0.19,
                'min_ratio': 0.0,
                'rep_len': 5,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'stopwords_filter': {
                'lang': 'en',
                'min_ratio': 0.3,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'flagged_words_filter': {
                'lang': 'en',
                'max_ratio': 0.01,
                'text_key': 'text',
                'tokenization': False,
                'use_words_aug': False,
                'words_aug_group_sizes': [2],
                'words_aug_join_char': ''
            }
        }, {
            'maximum_line_length_filter': {
                'min_len': 20,
                'text_key': 'text'
            }
        }, {
            'perplexity_filter': {
                'lang': 'en',
                'max_ppl': 1500,
                'text_key': 'text'
            }
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        target_process = [{
            'language_id_score_filter': {
                'lang': 'en',
                'min_score': 0.8,
                'text_key': 'text'
            }
        }, {
            'whitespace_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'punctuation_normalization_mapper': {
                'text_key': 'text'
            }
        }, {
            'fix_unicode_mapper': {
                'text_key': 'text'
            }
        }, {
            'remove_words_with_incorrect_substrings_mapper': {
                'lang': 'en',
                'substrings': None,
                'text_key': 'text',
                'tokenization': False
            }
        }, {
            'remove_long_words_mapper': {
                'max_len': 25,
                'min_len': 1,
                'text_key': 'text'
            }
        }, {
            'character_repetition_filter': {
                'max_ratio': 0.106,
                'min_ratio': 0.0,
                'rep_len': 10,
                'text_key': 'text'
            }
        }, {
            'special_characters_filter': {
                'max_ratio': 0.4,
                'min_ratio': 0.0,
                'text_key': 'text'
            }
        }, {
            'OpFusion:(average_line_length_filter,maximum_line_length_filter)': [{  # noqa: E501
                'average_line_length_filter': {
                    'min_len': 10,
                    'text_key': 'text',
                }
            }, {
                'maximum_line_length_filter': {
                    'min_len': 20,
                    'text_key': 'text',
                }
            }]
        }, {
            'OpFusion:(words_num_filter,word_repetition_filter,stopwords_filter,flagged_words_filter,perplexity_filter)': [{  # noqa: E501
                'words_num_filter': {
                    'lang': 'en',
                    'max_num': 100000,
                    'min_num': 20,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'word_repetition_filter': {
                    'lang': 'en',
                    'max_ratio': 0.19,
                    'min_ratio': 0.0,
                    'rep_len': 5,
                    'text_key': 'text',
                    'tokenization': False
                }
            }, {
                'stopwords_filter': {
                    'lang': 'en',
                    'min_ratio': 0.3,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'flagged_words_filter': {
                    'lang': 'en',
                    'max_ratio': 0.01,
                    'text_key': 'text',
                    'tokenization': False,
                    'use_words_aug': False,
                    'words_aug_group_sizes': [2],
                    'words_aug_join_char': ''
                }
            }, {
                'perplexity_filter': {
                    'lang': 'en',
                    'max_ppl': 1500,
                    'text_key': 'text'
                }
            }]
        }, {
            'document_simhash_deduplicator': {
                'hamming_distance': 4,
                'ignore_pattern': '\\p{P}',
                'lowercase': True,
                'num_blocks': 6,
                'text_key': 'text',
                'tokenization': 'space',
                'window_size': 6
            }
        }]
        self._run_op_fusion(original_process, target_process)


if __name__ == '__main__':
    unittest.main()
