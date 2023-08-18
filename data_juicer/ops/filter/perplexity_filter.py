# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from jsonargparse.typing import PositiveFloat

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import MODEL_ZOO, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import INTER_WORDS
from ..common import get_words_from_document


@OPERATORS.register_module('perplexity_filter')
@INTER_WORDS.register_module('perplexity_filter')
class PerplexityFilter(Filter):
    """Filter to keep samples with perplexity score less than a specific max
    value."""

    def __init__(self,
                 lang: str = 'en',
                 max_ppl: PositiveFloat = 1500,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: Compute perplexity for samples in which language.
        :param max_ppl: The max filter perplexity in this op, samples
            will be filtered if their perplexity exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.max_ppl = max_ppl
        self.sp_model_key = prepare_model(lang=lang,
                                          model_type='sentencepiece')
        self.kl_model_key = prepare_model(lang=lang, model_type='kenlm')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.perplexity in sample[Fields.stats]:
            return sample

        # tokenization
        words_key = f'words-{self.sp_model_key}'
        if context and words_key in sample['__dj__context__']:
            words = sample['__dj__context__'][words_key]
        else:
            tokenizer = MODEL_ZOO.get(self.sp_model_key, None)
            words = get_words_from_document(
                sample[self.text_key],
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            if context:
                sample['__dj__context__'][words_key] = words
        text = ' '.join(words)
        # compute perplexity
        logits, length = 0, 0
        kenlm_model = MODEL_ZOO.get(self.kl_model_key, None)
        for line in text.splitlines():
            logits += kenlm_model.score(line)
            length += (len(line.split()) + 1)
        ppl = (10.0**(-logits / length)) if length != 0 else 0.0
        sample[Fields.stats][StatsKeys.perplexity] = round(ppl, 1)

        return sample

    def process(self, sample):
        return sample[Fields.stats][StatsKeys.perplexity] <= self.max_ppl
