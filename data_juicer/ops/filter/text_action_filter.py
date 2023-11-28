import numpy as np
import string

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import prepare_model, get_model
from data_juicer.utils.mm_utils import SpecialTokens

from ..base_op import OPERATORS, Filter

OP_NAME = 'text_action_filter'

@OPERATORS.register_module(OP_NAME)
class TextActionFilter(Filter):
    """
    Filter to keep texts those contain actions in the text.
    """

    def __init__(self,
                 lang: str='en',
                 min_action_num: int=1,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: language of the text in the samples. 'en' for detection of 
            actions in English an'zh' for detection of actions in Chinese.
        :param mini_action_num: The min action number in the filtering. samples
            will be filtered if their action number in the text is below this
            parameter.
        """
        super().__init__(*args, **kwargs)

        if lang not in ['en', 'zh']:
            raise ValueError(f'Language [{lang}] is not supported in action detection. '
                             f'Can only be one of ["en", "zh"].')
        self.lang = lang
        self.model_key = prepare_model(model_type='spacy', lang=lang)
        self.action_poss = ['VERB']
        self.action_tags = ['VV', 'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN']
        self.min_action_num = min_action_num

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.num_action in sample[Fields.stats]:
            return sample

        # load text.
        special_token_dict = {
            key: value
            for key, value in SpecialTokens.__dict__.items()
            if not key.startswith('__')
        }
        def remove_special_token(text):
            for value in special_token_dict.values():
                text = text.replace(value, '')
            return text
        text = remove_special_token(sample[self.text_key])
            
        # process text via spacy and count the actions in text
        model = get_model(self.model_key)
        doc = model(text)
        num_action = 0
        for token in doc:
            if token.pos_ in self.action_poss and token.tag_ in self.action_tags:
                num_action += 1
        sample[Fields.stats][StatsKeys.num_action] = num_action

        return sample

    def process(self, sample):
        num_action = sample[Fields.stats][StatsKeys.num_action]
        if self.min_action_num <= num_action:
            return True
        else:
            return False
