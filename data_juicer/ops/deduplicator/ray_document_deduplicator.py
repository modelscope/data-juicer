import hashlib
import string

import regex as re

from ..base_op import OPERATORS
from .ray_basic_deduplicator import RayBasicDeduplicator

OP_NAME = "ray_document_deduplicator"


@OPERATORS.register_module(OP_NAME)
class RayDocumentDeduplicator(RayBasicDeduplicator):
    """Deduplicates samples at the document level using exact matching.

    This operator calculates a hash for each document and uses it to identify and remove
    duplicate documents. The hash is computed based on the text content of the document,
    with options to convert the text to lowercase and to ignore non-alphabet characters
    (whitespaces, digits, and punctuation). The deduplication process can use either a Ray
    actor or Redis as the backend for managing the hashes. The key metric used is the exact
    match of the document text, which is character-based by default."""

    def __init__(
        self,
        backend: str = "ray_actor",
        redis_address: str = "redis://localhost:6379",
        lowercase: bool = False,
        ignore_non_character: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param lowercase: Whether to convert sample text to lower case
        :param ignore_non_character: Whether to ignore non-alphabet
        characters, including whitespaces, digits, and punctuations
        :param args: extra args
        :param kwargs: extra args.
        """
        super().__init__(backend=backend, redis_address=redis_address, *args, **kwargs)
        self.lowercase = lowercase
        self.remove_non_character_regex = (
            re.compile(f"\s+|\d+|[{re.escape(string.punctuation)}]") if ignore_non_character else None  # noqa: W605
        )

    def calculate_hash(self, sample, context=False):
        if self.text_key not in sample or not sample[self.text_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

        text = sample[self.text_key]
        if self.lowercase:
            text = text.lower()
        if self.remove_non_character_regex:
            text = self.remove_non_character_regex.sub("", text)

        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()
