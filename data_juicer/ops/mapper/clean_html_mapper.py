# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/
# --------------------------------------------------------

from selectolax.parser import HTMLParser

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('clean_html_mapper')
class CleanHtmlMapper(Mapper):
    """Mapper to clean html code in text samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

    def process(self, sample):

        def _clean_html(raw_html):
            raw_html = raw_html.replace('<li>', '\n*')
            raw_html = raw_html.replace('</li>', '')
            raw_html = raw_html.replace('<ol>', '\n*')
            raw_html = raw_html.replace('</ol>', '')
            parser = HTMLParser(raw_html)
            return parser.text()

        sample[self.text_key] = _clean_html(sample[self.text_key])
        return sample
