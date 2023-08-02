from data_juicer.utils.model_utils import MODEL_ZOO, prepare_model

from ..base_op import OPERATORS, Mapper
from ..common import get_sentences_from_document


@OPERATORS.register_module('sentence_split_mapper')
class SentenceSplitMapper(Mapper):
    """Mapper to split text samples to sentences."""

    def __init__(self, lang: str = 'en', *args, **kwargs):
        """
        Initialization method.

        :param lang: split sentence of text in which language.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(lang=lang, model_type='nltk')

    def process(self, sample):

        nltk_model = MODEL_ZOO.get(self.model_key, None)
        sample[self.text_key] = get_sentences_from_document(
            sample[self.text_key],
            model_func=nltk_model.tokenize if nltk_model else None)
        return sample
