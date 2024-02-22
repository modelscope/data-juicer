import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_non_chinese_character_mapper')
class RemoveNonChineseCharacterlMapper(Mapper):
    """Mapper to remove non chinese Character in text samples."""

    def __init__(self,
                 keep_alphabet: bool = True,
                 keep_number: bool = True,
                 keep_punc: bool = True,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param keep_alphabet: whether to keep alphabet
        :param keep_number: whether to keep number
        :param keep_punc: whether to keep punctuation
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = u'[^\u4e00-\u9fa5'
        if keep_alphabet:
            self.pattern += u'A-Za-z'
        if keep_number:
            self.pattern += u'0-9'
        if keep_punc:
            self.pattern += u'.， ,\\-。%《*》/•、&＆(—)（+）：？!！“”·]+'
        else:
            self.pattern += u']'

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
