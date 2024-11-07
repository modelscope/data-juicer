import json
import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.model_utils import get_model, prepare_model

from ..common import split_text_by_punctuation

OP_NAME = 'extract_event_mapper'


# TODO: LLM-based inference.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractEventMapper(Mapper):
    """
    Extract events and relavant characters in the text
    """

    DEFAULT_SYSTEM_PROMPT = ('给定一段文本，对文本的情节进行分点总结，并抽取与情节相关的人物。\n'
                             '要求：\n'
                             '- 尽量不要遗漏内容，不要添加文本中没有的情节，符合原文事实\n'
                             '- 联系上下文说明前因后果，但仍然需要符合事实\n'
                             '- 不要包含主观看法\n'
                             '- 注意要尽可能保留文本的专有名词\n'
                             '- 注意相关人物需要在对应情节中出现\n'
                             '- 只抽取情节中的主要人物，不要遗漏情节的主要人物\n'
                             '- 总结格式如下：\n'
                             '### 情节1：\n'
                             '- **情节描述**： ...\n'
                             '- **相关人物**：人物1，人物2，人物3，...\n'
                             '### 情节2：\n'
                             '- **情节描述**： ...\n'
                             '- **相关人物**：人物1，人物2，...\n'
                             '### 情节3：\n'
                             '- **情节描述**： ...\n'
                             '- **相关人物**：人物1，...\n'
                             '...\n')
    DEFAULT_INPUT_TEMPLATE = '# 文本\n```\n{text}\n```\n'
    DEFAULT_OUTPUT_PATTERN = r"""
        \#\#\#\s*情节(\d+)：\s*
        -\s*\*\*情节描述\*\*\s*：\s*(.*?)\s*
        -\s*\*\*相关人物\*\*\s*：\s*(.*?)(?=\#\#\#|\Z)
    """

    def __init__(self,
                 api_model: str = 'gpt-4o',
                 *,
                 event_desc_key: str = Fields.event_description,
                 relavant_char_key: str = Fields.relavant_characters,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 response_path: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 input_template: Optional[str] = None,
                 output_pattern: Optional[str] = None,
                 try_num: PositiveInt = 3,
                 api_params: Optional[Dict] = None,
                 **kwargs):
        """
        Initialization method.
        :param api_model: API model name.
        :param event_desc_key: the field name to store the event descriptions
            in response. It's "__dj__event_description__" in default.
        :param relavant_char_key: the field name to store the relavant
            characters to the events in response.
            It's "__dj__relavant_characters__" in default.
        :param api_url: API URL. Defaults to DJ_API_URL environment variable.
        :param api_key: API key. Defaults to DJ_API_KEY environment variable.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the calibration task.
        :param input_template: Template for building the model input.
        :param output_pattern: Regular expression for parsing model output.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param api_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.event_desc_key = event_desc_key
        self.relavant_char_key = relavant_char_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.api_params = api_params or {}
        self.model_key = prepare_model(model_type='api',
                                       api_model=api_model,
                                       api_url=api_url,
                                       api_key=api_key,
                                       response_path=response_path)

        self.try_num = try_num

    def parse_output(self, raw_output):
        pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)

        contents = []
        for match in matches:
            _, description, characters = match
            contents.append({
                self.event_desc_key:
                description.strip(),
                self.relavant_char_key:
                split_text_by_punctuation(characters)
            })

        return contents

    def process_single(self, sample=None, rank=None):
        client = get_model(self.model_key, rank=rank)

        input_prompt = self.input_template.format(text=sample[self.text_key])
        messages = [{
            'role': 'system',
            'content': self.system_prompt
        }, {
            'role': 'user',
            'content': input_prompt
        }]

        contents = []
        for i in range(self.try_num):
            try:
                output = client(messages, **self.api_params)
                contents = self.parse_output(output)
                if len(contents) > 0:
                    break
            except Exception as e:
                logger.warning(f'Exception: {e}')

        sample[self.response_key] = json.dumps(contents)

        return sample
