from typing import List, Union

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_specific_chars_mapper')
class RemoveSpecificCharsMapper(Mapper):
    """Mapper to clean specific chars in text samples."""

    def __init__(self,
                 chars_to_remove: Union[str, List[str]] = '◆●■►▼▲▴∆▻▷❖♡□',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param chars_to_remove: a list or a string including all
            characters that need to be removed from text.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        if chars_to_remove:
            self.pattern = '[' + '|'.join(chars_to_remove) + ']'
        else:
            self.pattern = None

    def process(self, sample):

        if self.pattern is None:
            return sample

        sample[self.text_key] = re.sub(pattern=self.pattern,
                                       repl=r'',
                                       string=sample[self.text_key],
                                       flags=re.DOTALL)
        return sample
