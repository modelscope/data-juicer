import ftfy

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('fix_unicode_mapper')
class FixUnicodeMapper(Mapper):
    """Mapper to fix unicode errors in text samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process(self, sample):
        sample[self.text_key] = ftfy.fix_text(sample[self.text_key])
        return sample
