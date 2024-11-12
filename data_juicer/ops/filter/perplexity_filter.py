# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import get_words_from_document
from ..op_fusion import INTER_WORDS

OP_NAME = 'perplexity_filter'


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class PerplexityFilter(Filter):
    """Filter to keep samples with perplexity score less than a specific max
    value."""

    _batched_op = True

    def __init__(self,
                 lang: str = 'en',
                 max_ppl: float = 1500,
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
        self.sp_model_key = prepare_model(model_type='sentencepiece',
                                          lang=lang)
        self.kl_model_key = prepare_model(model_type='kenlm', lang=lang)

    def compute_stats_batched(self, samples, context=False):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        words_key = f'{InterVars.words}-{self.sp_model_key}'

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.perplexity in stat:
                continue
            # tokenization
            if context and words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][words_key]
            else:
                tokenizer = get_model(self.sp_model_key)
                words = get_words_from_document(
                    samples_list[idx],
                    token_func=tokenizer.encode_as_pieces
                    if tokenizer else None)
                if context:
                    samples[Fields.context][idx][words_key] = words
            text = ' '.join(words)
            # compute perplexity
            logits, length = 0, 0
            kenlm_model = get_model(self.kl_model_key)
            for line in text.splitlines():
                logits += kenlm_model.score(line)
                length += (len(line.split()) + 1)
            ppl = (10.0**(-logits / length)) if length != 0 else 0.0
            samples_stats[idx][StatsKeys.perplexity] = round(ppl, 1)

        return samples

    def process_batched(self, samples):
        if isinstance(samples[Fields.stats], list):
            return map(lambda stat: stat[StatsKeys.perplexity] <= self.max_ppl,
                       samples[Fields.stats])
        else:
            return samples[Fields.stats][StatsKeys.perplexity] <= self.max_ppl
