import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("remove_non_chinese_character_mapper")
class RemoveNonChineseCharacterlMapper(Mapper):
    """Mapper to remove non chinese Character in text samples."""

    _batched_op = True

    def __init__(self, keep_alphabet: bool = True, keep_number: bool = True, keep_punc: bool = True, *args, **kwargs):
        """
        Initialization method.

        :param keep_alphabet: whether to keep alphabet
        :param keep_number: whether to keep number
        :param keep_punc: whether to keep punctuation
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = "[^\u4e00-\u9fa5"
        if keep_alphabet:
            self.pattern += "A-Za-z"
        if keep_number:
            self.pattern += "0-9"
        if keep_punc:
            self.pattern += ".， ,\\-。%《*》/•、&＆(—)（+）：？!！“”·]+"
        else:
            self.pattern += "]"

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            if not re.search(self.pattern, text, flags=re.DOTALL):
                continue

            samples[self.text_key][idx] = re.sub(pattern=self.pattern, repl=r"", string=text, flags=re.DOTALL)
        return samples
