import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('clean_ip_mapper')
class CleanIpMapper(Mapper):
    """Mapper to clean ipv4 and ipv6 address in text samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        self.pattern = r'(?:(?:1[0-9][0-9]\.)|(?:2[0-4][0-9]\.)|'
        self.pattern += r'(?:25[0-5]\.)|(?:[1-9][0-9]\.)|(?:[0-9]\.))'
        self.pattern += r'{3}(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|'
        self.pattern += r'(?:25[0-5])|(?:[1-9][0-9])|(?:[0-9]))|'
        self.pattern += r'([\da-fA-F]{1,4}:){7}[\da-fA-F]{1,4}'  # ipv6

    def process(self, sample):

        if not re.search(self.pattern, sample[self.text_key], flags=re.DOTALL):
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
