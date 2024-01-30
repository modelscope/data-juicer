from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..common import get_sentences_from_document

OP_NAME = 'sentence_split_mapper'

with AvailabilityChecking(['nltk'], OP_NAME):
    import nltk  # noqa: F401


@OPERATORS.register_module(OP_NAME)
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
        self.lang = lang
        self.model_key = prepare_model(model_type='nltk', lang=lang)

    def process(self, sample):

        nltk_model = get_model(self.model_key)
        sample[self.text_key] = get_sentences_from_document(
            sample[self.text_key],
            model_func=nltk_model.tokenize if nltk_model else None)
        return sample
