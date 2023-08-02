# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/
# --------------------------------------------------------

from typing import List, Union

import regex as re

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module('remove_comments_mapper')
class RemoveCommentsMapper(Mapper):
    """
    Mapper to remove comments in different kinds of documents.

    Only support 'tex' \ for now.
    """

    def __init__(self,
                 doc_type: Union[str, List[str]] = 'tex',
                 inline: bool = True,
                 multiline: bool = True,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param doc_type: Type of document to remove comments.
        :param inline: Whether to remove inline comments.
        :param multiline: Whether to remove multiline comments.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.doc_type = doc_type
        self.inline = inline
        self.multiline = multiline

    def process(self, sample):
        # TODO: remove different comments by sample type

        if self.inline:
            # remove all in comments within a line
            sample[self.text_key] = re.sub(pattern=r'[^\\]%.+$',
                                           repl=r'',
                                           string=sample[self.text_key],
                                           flags=re.MULTILINE)

        if self.multiline:
            sample[self.text_key] = re.sub(pattern=r'(?m)^%.*\n?',
                                           repl=r'',
                                           string=sample[self.text_key],
                                           flags=re.MULTILINE)
        return sample
