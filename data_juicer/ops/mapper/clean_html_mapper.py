# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

import lazy_loader as lazy

from ..base_op import AUTOINSTALL, OPERATORS, Mapper

OP_NAME = 'clean_html_mapper'

selectolax = lazy.load('selectolax')


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
        AUTOINSTALL.check(['selectolax'])

    def process(self, samples):

        def _clean_html(raw_html):
            raw_html = raw_html.replace('<li>', '\n*')
            raw_html = raw_html.replace('</li>', '')
            raw_html = raw_html.replace('<ol>', '\n*')
            raw_html = raw_html.replace('</ol>', '')
            parser = selectolax.parser.HTMLParser(raw_html)
            return parser.text()

        samples[self.text_key] = list(
            map(lambda text: _clean_html(text), samples[self.text_key]))
        return samples
