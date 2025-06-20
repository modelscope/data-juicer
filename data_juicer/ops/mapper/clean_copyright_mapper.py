# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("clean_copyright_mapper")
class CleanCopyrightMapper(Mapper):
    """Mapper to clean copyright comments at the beginning of the text
    samples."""

    _batched_op = True

    def __init__(self, *args, **kwargs):
        """
        Initialization method.

        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.pat = re.compile("/\\*[^*]*\\*+(?:[^/*][^*]*\\*+)*/")
        self.cpat = re.compile("copyright", re.IGNORECASE)

    def _process_single_sample(self, sample):
        r = self.pat.search(sample)
        if r:
            # found one, now see if it contains "copyright", if so strip it
            span = r.span()
            sub = sample[span[0] : span[1]]
            if self.cpat.search(sub):
                # cut it
                sample = sample[: span[0]] + sample[span[1] :]

            return sample

        lines = sample.split("\n")
        skip = 0

        # Greedy replace any file that begins with comment block, most
        # are copyright headers
        for k in range(len(lines)):
            if lines[k].startswith("//") or lines[k].startswith("#") or lines[k].startswith("--") or not lines[k]:
                skip = skip + 1
            else:
                break

        if skip:
            # we skipped, consume it
            sample = "\n".join(lines[skip:])
        return sample

    def process_batched(self, samples):
        samples[self.text_key] = [self._process_single_sample(text) for text in samples[self.text_key]]
        return samples
