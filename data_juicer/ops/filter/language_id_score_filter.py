from jsonargparse.typing import ClosedUnitInterval
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import prepare_model, get_model

from ..base_op import OPERATORS, Filter


@OPERATORS.register_module('language_id_score_filter')
class LanguageIDScoreFilter(Filter):
    """Filter to keep samples in a specific language with confidence score
    larger than a specific min value."""

    def __init__(self,
                 lang: str = '',
                 min_score: ClosedUnitInterval = 0.8,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: Samples in which language to keep.
        :param min_score: The min language identification confidence
            scores of samples to keep.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.min_score = min_score
        self.model_key = prepare_model(lang=lang, model_type='fasttext')

    def compute_stats(self, sample):
        # check if it's computed already
        if StatsKeys.lang in sample[
                Fields.stats] and StatsKeys.lang_score in sample[Fields.stats]:
            return sample

        text = sample[self.text_key].lower().replace('\n', ' ')
        ft_model = get_model(self.model_key, lang=self.lang, model_type='fasttext')
        if ft_model is None:
            err_msg = 'Model not loaded. Please retry later.'
            logger.error(err_msg)
            raise ValueError(err_msg)
        pred = ft_model.predict(text)
        lang_id = pred[0][0].replace('__label__', '')
        lang_score = pred[1][0]

        sample[Fields.stats][StatsKeys.lang] = lang_id
        sample[Fields.stats][StatsKeys.lang_score] = lang_score

        return sample

    def process(self, sample):
        if self.lang:
            return sample[Fields.stats][StatsKeys.lang] == self.lang \
                   and sample[Fields.stats][StatsKeys.lang_score] >= self.min_score
        else:
            return sample[Fields.stats][StatsKeys.lang_score] >= self.min_score
