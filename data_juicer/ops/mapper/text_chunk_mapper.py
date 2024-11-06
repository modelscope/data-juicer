import re
from itertools import chain
from typing import Union

from pydantic import NonNegativeInt, PositiveInt

from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

OP_NAME = 'text_chunk_mapper'


@OPERATORS.register_module(OP_NAME)
class TextChunkMapper(Mapper):
    """Split input text to chunks."""

    _batched_op = True

    def __init__(self,
                 max_len: Union[PositiveInt, None] = None,
                 split_pattern: Union[str, None] = r'\n\n',
                 overlap_len: NonNegativeInt = 0,
                 hf_tokenizer: Union[str, None] = None,
                 trust_remote_code: bool = False,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param max_len: Split text into multi texts with this max len if not
            None.
        :param split_pattern: Make sure split in this pattern if it is not None
            and force cut if the length exceeds max_len.
        :param overlap_len: Overlap length of the split texts if not split in
            the split pattern.
        :param hf_tokenizer: The tokenizer name of Hugging Face tokenizers.
            The text length will be calculate as the token num if it is offerd.
            Otherwise, the text length equals to string length.
        :trust_remote_code: for loading huggingface model
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        if max_len is None and split_pattern is None:
            raise ValueError('max_len and split_pattern cannot be both None')

        if max_len is not None and overlap_len >= max_len:
            raise ValueError('overlap_len must be less than max_len')

        self.max_len = max_len
        self.overlap_len = overlap_len
        self.split_pattern = split_pattern
        self.hf_tokenizer = hf_tokenizer
        if hf_tokenizer is not None:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_tokenizer,
                return_model=False,
                trust_remote_code=trust_remote_code)

    def recursively_chunk(self, text):
        if self.hf_tokenizer is not None:
            tokenizer = get_model(self.model_key)
            tokens = tokenizer.tokenize(text)
            tokens = [t.decode(encoding='UTF-8') for t in tokens]
            total_len = len(tokens)
            sub_text = ''.join(tokens[:self.max_len])
        else:
            total_len = len(text)
            sub_text = text[:self.max_len]

        if total_len <= self.max_len:
            return [text]

        matches = list(re.finditer(self.split_pattern, sub_text))
        if not matches:
            cur_text = sub_text
            if self.hf_tokenizer is not None:
                left_text = ''.join(tokens[self.max_len - self.overlap_len:])
            else:
                left_text = text[self.max_len - self.overlap_len:]
        else:
            last_match = matches[-1]
            cur_text = sub_text[:last_match.start()]
            left_text = text[last_match.end():]

        return [cur_text] + self.recursively_chunk(left_text)

    def get_text_chunks(self, text):

        if self.split_pattern is not None and self.max_len is None:
            chunks = re.split(f'({self.split_pattern})', text)
            chunks = [t for t in chunks if t.strip()]
        elif self.split_pattern is None and self.max_len is not None:
            tokens = text
            total_len = len(text)
            if self.hf_tokenizer is not None:
                tokenizer = get_model(self.model_key)
                tokens = tokenizer.tokenize(text)
                tokens = [t.decode(encoding='UTF-8') for t in tokens]
                total_len = len(tokens)
            if total_len <= self.max_len:
                return [text]
            chunks = []
            for start in range(0, total_len, self.max_len - self.overlap_len):
                cur = tokens[start:start + self.max_len]
                if self.hf_tokenizer is not None:
                    cur = ''.join(cur)
                chunks.append(cur)
        else:
            chunks = self.recursively_chunk(text)

        return chunks

    def process_batched(self, samples):

        sample_num = len(samples[self.text_key])

        samples[self.text_key] = [
            self.get_text_chunks(text) for text in samples[self.text_key]
        ]

        for key in samples:
            if key != self.text_key:
                samples[key] = [[samples[key][i]] *
                                len(samples[self.text_key][i])
                                for i in range(len(sample_num))]

        for key in samples:
            samples[key] = list(chain(*samples[key]))

        return samples
