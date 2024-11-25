import re
from itertools import chain
from typing import Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'extract_entity_attribute_mapper'


# TODO: LLM-based inference.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractEntityAttributeMapper(Mapper):
    """
    Extract attributes for given entities from the text
    """

    _batched_op = True

    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        '给定一段文本，从文本中总结{entity}的{attribute}，并且从原文摘录最能说明该{attribute}的代表性示例。\n'
        '要求：\n'
        '- 摘录的示例应该简短。\n'
        '- 遵循如下的回复格式：\n'
        '## {attribute}：\n'
        '{entity}的{attribute}描述...\n'
        '### 代表性示例1：\n'
        '说明{entity}该{attribute}的原文摘录1...\n'
        '### 代表性示例2：\n'
        '说明{entity}该{attribute}的原文摘录2...\n'
        '...\n')

    DEFAULT_INPUT_TEMPLATE = '# 文本\n```\n{text}\n```\n'
    DEFAULT_ATTR_PATTERN_TEMPLATE = r'\#\#\s*{attribute}：\s*(.*?)(?=\#\#\#|\Z)'
    DEFAULT_DEMON_PATTERN = r'\#\#\#\s*代表性示例(\d+)：\s*(.*?)(?=\#\#\#|\Z)'

    def __init__(self,
                 query_entities: List[str] = [],
                 query_attributes: List[str] = [],
                 api_model: str = 'gpt-4o',
                 *,
                 entity_key: str = Fields.main_entity,
                 attribute_key: str = Fields.attribute,
                 attribute_desc_key: str = Fields.attribute_description,
                 support_text_key: str = Fields.attribute_support_text,
                 api_endpoint: Optional[str] = None,
                 response_path: Optional[str] = None,
                 system_prompt_template: Optional[str] = None,
                 input_template: Optional[str] = None,
                 attr_pattern_template: Optional[str] = None,
                 demo_pattern: Optional[str] = None,
                 try_num: PositiveInt = 3,
                 drop_text: bool = False,
                 model_params: Dict = {},
                 sampling_params: Dict = {},
                 **kwargs):
        """
        Initialization method.
        :param query_entities: Entity list to be queried.
        :param query_attributes: Attribute list to be queried.
        :param api_model: API model name.
        :param entity_key: The field name to store the given main entity for
            attribute extraction. It's "__dj__entity__" in default.
        :param entity_attribute_key: The field name to store the given
            attribute to be extracted. It's "__dj__attribute__" in default.
        :param attribute_desc_key: The field name to store the extracted
            attribute description. It's "__dj__attribute_description__" in
            default.
        :param support_text_key: The field name to store the attribute
            support text extracted from the raw text. It's
            "__dj__support_text__" in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt_template: System prompt template for the
            task. Need to be specified by given entity and attribute.
        :param input_template: Template for building the model input.
        :param attr_pattern_template: Pattern for parsing the attribute from
            output. Need to be specified by given attribute.
        :param: demo_pattern: Pattern for parsing the demonstraction from
            output to support the attribute.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.query_entities = query_entities
        self.query_attributes = query_attributes

        self.entity_key = entity_key
        self.attribute_key = attribute_key
        self.attribute_desc_key = attribute_desc_key
        self.support_text_key = support_text_key

        self.system_prompt_template = system_prompt_template \
            or self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.attr_pattern_template = attr_pattern_template \
            or self.DEFAULT_ATTR_PATTERN_TEMPLATE
        self.demo_pattern = demo_pattern or self.DEFAULT_DEMON_PATTERN

        self.sampling_params = sampling_params
        self.model_key = prepare_model(model_type='api',
                                       model=api_model,
                                       endpoint=api_endpoint,
                                       response_path=response_path,
                                       **model_params)

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output, attribute_name):

        attribute_pattern = self.attr_pattern_template.format(
            attribute=attribute_name)
        pattern = re.compile(attribute_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)
        if matches:
            attribute = matches[0].strip()
        else:
            attribute = ''

        pattern = re.compile(self.demo_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)
        demos = [demo.strip() for _, demo in matches if demo.strip()]

        return attribute, demos

    def _process_single_sample(self, text='', rank=None):
        client = get_model(self.model_key, rank=rank)

        entities, attributes, descs, demo_lists = [], [], [], []
        for entity in self.query_entities:
            for attribute in self.query_attributes:
                system_prompt = self.system_prompt_template.format(
                    entity=entity, attribute=attribute)
                input_prompt = self.input_template.format(text=text)
                messages = [{
                    'role': 'system',
                    'content': system_prompt
                }, {
                    'role': 'user',
                    'content': input_prompt
                }]

                desc, demos = '', []
                for i in range(self.try_num):
                    try:
                        output = client(messages, **self.sampling_params)
                        desc, demos = self.parse_output(output, attribute)
                        if desc and len(demos) > 0:
                            break
                    except Exception as e:
                        logger.warning(f'Exception: {e}')
                entities.append(entity)
                attributes.append(attribute)
                descs.append(desc)
                demo_lists.append(demos)

        return entities, attributes, descs, demo_lists

    def process_batched(self, samples, rank=None):

        sample_num = len(samples[self.text_key])

        entities, attributes, descs, demo_lists = [], [], [], []
        for text in samples[self.text_key]:
            res = self._process_single_sample(text, rank=rank)
            cur_ents, cur_attrs, cur_descs, cur_demos = res
            entities.append(cur_ents)
            attributes.append(cur_attrs)
            descs.append(cur_descs)
            demo_lists.append(cur_demos)

        if self.drop_text:
            samples.pop(self.text_key)

        for key in samples:
            samples[key] = [[samples[key][i]] * len(descs[i])
                            for i in range(sample_num)]
        samples[self.entity_key] = entities
        samples[self.attribute_key] = attributes
        samples[self.attribute_desc_key] = descs
        samples[self.support_text_key] = demo_lists

        for key in samples:
            samples[key] = list(chain(*samples[key]))

        return samples
