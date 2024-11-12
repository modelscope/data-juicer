import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import AUTOINSTALL
from data_juicer.utils.mm_utils import remove_special_tokens
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

OP_NAME = 'text_entity_dependency_filter'


@OPERATORS.register_module(OP_NAME)
class TextEntityDependencyFilter(Filter):
    """
    Identify the entities in the text which are independent with other token,
    and filter them. The text containing no entities will be omitted.
    """

    def __init__(self,
                 lang: str = 'en',
                 min_dependency_num: int = 1,
                 any_or_all: str = 'all',
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: language of the text in the samples. 'en' for detection of
            entities in English and 'zh' for detection of entities in Chinese.
        :param mini_dependency_num: The min token number in the filtering.
            Objects is independent if their number of edges in the dependency
            tree is below this parameter.
        :param any_or_all: keep this sample with 'any' or 'all' strategy.
            'any': keep this sample if any objet is dependent. 'all': keep this
            sample only if all images are dependent.
        """
        super().__init__(*args, **kwargs)
        # '--no-deps' do not update numpy
        AUTOINSTALL.check(['spacy-pkuseg'], '--no-deps')

        if lang not in ['en', 'zh']:
            raise ValueError(
                f'Language [{lang}] is not supported in entities detection.'
                f'Can only be one of ["en", "zh"].')
        self.lang = lang
        self.model_key = prepare_model(model_type='spacy', lang=lang)
        self.entity_poss = ['NOUN', 'PROPN', 'PRON']
        self.entity_tags = ['NN', 'NR', 'PN', 'NNS', 'NNP', 'NNPS', 'PRP']
        self.min_dependency_num = min_dependency_num
        if any_or_all not in ['any', 'all']:
            raise ValueError(f'Keep strategy [{any_or_all}] is not supported. '
                             f'Can only be one of ["any", "all"].')
        self.any = (any_or_all == 'any')

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.num_dependency_edges in sample[Fields.stats]:
            return sample

        text = remove_special_tokens(sample[self.text_key])

        # identify entities
        model = get_model(self.model_key)
        doc = model(text)
        entity_to_dependency_nums = {}
        for token in doc:
            if token.pos_ in self.entity_poss \
             and token.tag_ in self.entity_tags:
                entity_to_dependency_nums[token] = 0

        # count the edges of each entity in dependency tree
        for obj in entity_to_dependency_nums:
            if obj.dep_ != 'ROOT':
                entity_to_dependency_nums[obj] += 1
        for token in doc:
            # the punctation mark such as ',', '.'
            if token.pos_ == 'PUNCT':
                continue

            if token.head in entity_to_dependency_nums.keys(
            ) and token.dep_ != 'ROOT':
                entity_to_dependency_nums[token.head] += 1

        sample[Fields.stats][StatsKeys.num_dependency_edges] = [
            n for _, n in entity_to_dependency_nums.items()
        ]

        return sample

    def process_single(self, sample):
        num_dependency_edges = sample[Fields.stats][
            StatsKeys.num_dependency_edges]
        keep_bools = np.array([
            self.min_dependency_num <= num_edge
            for num_edge in num_dependency_edges
        ])
        # omit the samples without entity
        if len(keep_bools) <= 0:
            return False

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
