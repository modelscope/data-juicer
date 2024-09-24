# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

from data_juicer.utils.availability_utils import AvailabilityChecking

from ..base_op import OPERATORS, Mapper

OP_NAME = 'clean_html_mapper'

with AvailabilityChecking(['selectolax'], OP_NAME):
    from selectolax.parser import HTMLParser


@OPERATORS.register_module(OP_NAME)
class CleanHtmlMapper(Mapper):
    """Mapper to clean html code in text samples."""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process(self, samples):

        def _clean_html(raw_html):
            raw_html = raw_html.replace('<li>', '\n*')
            raw_html = raw_html.replace('</li>', '')
            raw_html = raw_html.replace('<ol>', '\n*')
            raw_html = raw_html.replace('</ol>', '')
            parser = HTMLParser(raw_html)
            return parser.text()

        samples[self.text_key] = [
            _clean_html(text) for text in samples[self.text_key]
        ]
        return samples
