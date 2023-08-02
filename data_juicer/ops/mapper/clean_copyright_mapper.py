# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('clean_copyright_mapper')
class CleanCopyrightMapper(Mapper):
    """Mapper to clean copyright comments at the beginning of the text
    samples."""

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pat = re.compile('/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/')
        self.cpat = re.compile('copyright', re.IGNORECASE)

    def process(self, sample):

        r = self.pat.search(sample[self.text_key])
        if r:
            # found one, now see if it contains "copyright", if so strip it
            span = r.span()
            sub = sample[self.text_key][span[0]:span[1]]
            if self.cpat.search(sub):
                # cut it
                sample[self.text_key] = sample[
                    self.text_key][:span[0]] + sample[self.text_key][span[1]:]

            return sample

        lines = sample[self.text_key].split('\n')
        skip = 0

        # Greedy replace any file that begins with comment block, most
        # are copyright headers
        for k in range(len(lines)):
            if (lines[k].startswith('//') or lines[k].startswith('#')
                    or lines[k].startswith('--') or not lines[k]):
                skip = skip + 1
            else:
                break

        if skip:
            # we skipped, consume it
            sample[self.text_key] = '\n'.join(lines[skip:])
        return sample
