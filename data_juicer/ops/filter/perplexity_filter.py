# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from jsonargparse.typing import PositiveFloat

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import get_words_from_document
from ..op_fusion import INTER_WORDS

OP_NAME = 'perplexity_filter'

with AvailabilityChecking(['sentencepiece', 'kenlm'], OP_NAME):
    import kenlm  # noqa: F401
    import sentencepiece  # noqa: F401


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
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
        self.lang = lang
        self.sp_model_key = prepare_model(lang=lang,
                                          model_type='sentencepiece')
        self.kl_model_key = prepare_model(lang=lang, model_type='kenlm')

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.perplexity in sample[Fields.stats]:
            return sample

        # tokenization
        words_key = f'{InterVars.words}-{self.sp_model_key}'
        if context and words_key in sample[Fields.context]:
            words = sample[Fields.context][words_key]
        else:
            tokenizer = get_model(self.sp_model_key, self.lang,
                                  'sentencepiece')
            words = get_words_from_document(
                sample[self.text_key],
                token_func=tokenizer.encode_as_pieces if tokenizer else None)
            if context:
                sample[Fields.context][words_key] = words
        text = ' '.join(words)
        # compute perplexity
        logits, length = 0, 0
        kenlm_model = get_model(self.kl_model_key, self.lang, 'kenlm')
        for line in text.splitlines():
            logits += kenlm_model.score(line)
            length += (len(line.split()) + 1)
        ppl = (10.0**(-logits / length)) if length != 0 else 0.0
        sample[Fields.stats][StatsKeys.perplexity] = round(ppl, 1)

        return sample

    def process(self, sample):
        return sample[Fields.stats][StatsKeys.perplexity] <= self.max_ppl
