from data_juicer.utils.availability_utils import AvailabilityChecking

from ..base_op import OPERATORS, Mapper

OP_NAME = 'fix_unicode_mapper'

with AvailabilityChecking(['ftfy'], OP_NAME):
    import ftfy


@OPERATORS.register_module(OP_NAME)
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
