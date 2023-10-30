import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_non_chinese_character_mapper')
class RemoveNonChineseCharacterlMapper(Mapper):
    """Mapper to remove non chinese Character in text samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pattern = r'[^\u4e00-\u9fa5]'

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
