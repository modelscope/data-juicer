import hashlib
import string

import regex as re
from pydantic import PositiveInt

from ..base_op import OPERATORS
from .ray_basic_deduplicator import RayBasicDeduplicator

OP_NAME = 'ray_document_deduplicator'


@OPERATORS.register_module(OP_NAME)
class RayDocumentDeduplicator(RayBasicDeduplicator):
    """
    Deduplicator to deduplicate samples at document-level using exact matching.
    """

    def __init__(self,
                 redis_host: str = 'localhost',
                 redis_port: PositiveInt = 6380,
                 lowercase: bool = False,
                 ignore_non_character: bool = False,
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param redis_host: the hostname of redis server
        :param redis_port: the port of redis server
        :param lowercase: Whether to convert sample text to lower case
        :param ignore_non_character: Whether to ignore non-alphabet
        characters, including whitespaces, digits, and punctuations
        :param args: extra args
        :param kwargs: extra args.
        """
        super().__init__(redis_host=redis_host,
                         redis_port=redis_port,
                         *args,
                         **kwargs)
        self.lowercase = lowercase
        self.remove_non_character_regex = re.compile(
            f'\s+|\d+|[{re.escape(string.punctuation)}]'  # noqa: W605
        ) if ignore_non_character else None

    def calculate_hash(self, sample, context=False):
        if self.text_key not in sample or not sample[self.text_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

        text = sample[self.text_key]
        if self.lowercase:
            text = text.lower()
        if self.remove_non_character_regex:
            text = self.remove_non_character_regex.sub('', text)

        return hashlib.md5(text.strip().encode('utf-8')).hexdigest()
