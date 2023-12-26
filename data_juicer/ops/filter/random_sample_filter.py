import sys
import random  

from jsonargparse.typing import PositiveFloat  

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import get_words_from_document

@OPERATORS.register_module('random_sample_filter')
class RandomSampleFilter(Filter):
    """Filter to randomly sample a percentage of samples."""

    def __init__(self,
                 tokenization: bool = False,
                 sample_percentage: PositiveFloat = 0.1,  
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_tokenizer: the tokenizer name of Hugging Face tokenizers.
        :param sample_percentage: The percentage of samples to keep.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.sample_percentage = sample_percentage
        self.model_key = None
        

    def compute_stats(self, sample):
        
        return sample

    def process(self, sample):
        if random.uniform(0, 1) <= self.sample_percentage:
            return True
        else:
            return False
