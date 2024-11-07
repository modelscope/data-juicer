import json
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
    DEFAULT_ATTR_PATTERN_TEMPLATE = r'\#\#\s*\{attribute\}：\s*(?=\#\#\#|\Z)'
    DEFAULT_DEMON_PATTERN = r'\#\#\#\s*代表性示例(\d+)：\s*(?=\#\#\#|\Z)'

    def __init__(self,
                 query_entities: List[str],
                 query_attributes: List[str],
                 api_model: str = 'gpt-4o',
                 *,
                 entity_key: str = Fields.entity,
                 entity_attribute_key: str = Fields.entity_attribute,
                 support_text_key: str = Fields.support_text,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 response_path: Optional[str] = None,
                 system_prompt_template: Optional[str] = None,
                 input_template: Optional[str] = None,
                 attr_pattern_template: Optional[str] = None,
                 demo_pattern: Optional[str] = None,
                 try_num: PositiveInt = 3,
                 api_params: Optional[Dict] = None,
                 **kwargs):
        """
        Initialization method.
        :param query_entities: entity list to be queried.
        :param query_attributes: attribute list to be queried.
        :param api_model: API model name.
        :param entity_key: the field name to store the entity.
            It's "__dj__entity__" in default.
        :param entity_attribute_key: the field name to store the attribute.
            It's "__dj__entity_attribute__" in default.
        :param support_text_key: the field name to store the attribute
            support text extracted from the raw text. It's
            "__dj__support_text__" in default.
        :param api_url: API URL. Defaults to DJ_API_URL environment variable.
        :param api_key: API key. Defaults to DJ_API_KEY environment variable.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt_template: System prompt for the calibration
            task. Need to be specified by given entity and attribute.
        :param input_template: Template for building the model input.
        :param attr_pattern_template: Pattern for parsing the attribute from
            output. Need to be specified by given attribute.
        :param: demo_pattern: Pattern for parsing the demonstraction from
            output to support the attribute.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param api_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.query_entities = query_entities
        self.query_attributes = query_attributes

        self.entity_key = entity_key
        self.entity_attribute_key = entity_attribute_key
        self.support_text_key = support_text_key

        self.system_prompt_template = system_prompt_template \
            or self.DEFAULT_SYSTEM_PROMPT_TEMPLATE
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.attr_pattern_template = attr_pattern_template \
            or self.DEFAULT_ATTR_PATTERN_TEMPLATE
        self.demo_pattern = demo_pattern or self.DEFAULT_DEMON_PATTERN

        self.api_params = api_params or {}
        self.model_key = prepare_model(model_type='api',
                                       api_model=api_model,
                                       api_url=api_url,
                                       api_key=api_key,
                                       response_path=response_path)

        self.try_num = try_num

    def parse_output(self, raw_output, attribute_name):

        attribute_pattern = self.attr_pattern_template.format(
            attribute=attribute_name)
        print('attribute_pattern', attribute_pattern)
        pattern = re.compile(attribute_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)
        if matches:
            attribute = matches[0]
        else:
            attribute = ''

        pattern = re.compile(self.demo_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)

        demos = []
        for match in matches:
            _, demo = match
            demos.append(demo)

        return {attribute_name: attribute, self.support_text_key: demos}

    def _process_single_sample(self, text='', rank=None):
        client = get_model(self.model_key, rank=rank)

        results = []
        for entity in self.query_entities:
            for attribute in self.query_attributes:
                system_prompt = self.system_prompt_template.format(
                    entity=entity, attribute=attribute)
                print(system_prompt)
                input_prompt = self.input_template.format(text=text)
                print(input_prompt)
                messages = [{
                    'role': 'system',
                    'content': system_prompt
                }, {
                    'role': 'user',
                    'content': input_prompt
                }]

                result = {attribute: '', self.support_text_key: []}
                for i in range(self.try_num):
                    try:
                        output = client(messages, **self.api_params)
                        print(output)
                        result = self.parse_output(output, attribute)
                        if result[attribute]:
                            break
                    except Exception as e:
                        logger.warning(f'Exception: {e}')
                result = json.dumps({
                    self.entity_key: entity,
                    self.entity_attribute_key: result
                })
                print(result)

                results.append(result)

        return results

    def process_batched(self, samples):

        sample_num = len(samples[self.text_key])

        samples[self.response_key] = [
            self._process_single_sample(text)
            for text in samples[self.text_key]
        ]

        for key in samples:
            if key != self.response_key:
                samples[key] = [[samples[key][i]] *
                                len(samples[self.response_key][i])
                                for i in range(len(sample_num))]

        for key in samples:
            samples[key] = list(chain(*samples[key]))

        return samples
