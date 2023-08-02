# Some code here has been modified from:
# https://github.com/bigscience-workshop/data-preparation
# --------------------------------------------------------

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('punctuation_normalization_mapper')
class PunctuationNormalizationMapper(Mapper):
    """Mapper to normalize unicode punctuations to English punctuations in text
    \ samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.punctuation_unicode = {
            '，': ',',
            '。': '.',
            '、': ',',
            '„': '"',
            '”': '"',
            '“': '"',
            '«': '"',
            '»': '"',
            '１': '"',
            '」': '"',
            '「': '"',
            '《': '"',
            '》': '"',
            '´': "'",
            '∶': ':',
            '：': ':',
            '？': '?',
            '！': '!',
            '（': '(',
            '）': ')',
            '；': ';',
            '–': '-',
            '—': ' - ',
            '．': '. ',
            '～': '~',
            '’': "'",
            '…': '...',
            '━': '-',
            '〈': '<',
            '〉': '>',
            '【': '[',
            '】': ']',
            '％': '%',
            '►': '-',
        }

    def process(self, sample):
        sample[self.text_key] = ''.join([
            self.punctuation_unicode.get(c, c) for c in sample[self.text_key]
        ])
        return sample
