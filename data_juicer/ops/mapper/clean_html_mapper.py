# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

selectolax = LazyLoader("selectolax")

OP_NAME = "clean_html_mapper"


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

    def process_batched(self, samples):
        def _clean_html(raw_html):
            raw_html = raw_html.replace("<li>", "\n*")
            raw_html = raw_html.replace("</li>", "")
            raw_html = raw_html.replace("<ol>", "\n*")
            raw_html = raw_html.replace("</ol>", "")
            parser = selectolax.parser.HTMLParser(raw_html)
            return parser.text()

        samples[self.text_key] = [_clean_html(text) for text in samples[self.text_key]]
        return samples
