import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE, Mapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'extract_nickname_mapper'


# TODO: LLM-based inference.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ExtractNicknameMapper(Mapper):
    """
    Extract nickname relationship in the text
    """

    DEFAULT_SYSTEM_PROMPT = ('给定你一段文本，你的任务是将人物之间的称呼方式（昵称）提取出来。\n'
                             '要求：\n'
                             '- 需要给出说话人对被称呼人的称呼，不要搞反了。\n'
                             '- 相同的说话人和被称呼人最多给出一个最常用的称呼。\n'
                             '- 请不要输出互相没有昵称的称呼方式。\n'
                             '- 输出格式如下：\n'
                             '```\n'
                             '### 称呼方式1\n'
                             '- **说话人**：...\n'
                             '- **被称呼人**：...\n'
                             '- **...对...的昵称**：...\n'
                             '### 称呼方式2\n'
                             '- **说话人**：...\n'
                             '- **被称呼人**：...\n'
                             '- **...对...的昵称**：...\n'
                             '### 称呼方式3\n'
                             '- **说话人**：...\n'
                             '- **被称呼人**：...\n'
                             '- **...对...的昵称**：...\n'
                             '...\n'
                             '```\n')
    DEFAULT_INPUT_TEMPLATE = '# 文本\n```\n{text}\n```\n'
    DEFAULT_OUTPUT_PATTERN = r"""
        \#\#\#\s*称呼方式(\d+)\s*
        -\s*\*\*说话人\*\*\s*：\s*(.*?)\s*
        -\s*\*\*被称呼人\*\*\s*：\s*(.*?)\s*
        -\s*\*\*(.*?)对(.*?)的昵称\*\*\s*：\s*(.*?)(?=\#\#\#|\Z) # for double check
    """

    def __init__(self,
                 api_model: str = 'gpt-4o',
                 *,
                 nickname_key: str = Fields.nickname,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 response_path: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 input_template: Optional[str] = None,
                 output_pattern: Optional[str] = None,
                 try_num: PositiveInt = 3,
                 drop_text: bool = False,
                 model_params: Optional[Dict] = {},
                 sampling_params: Optional[Dict] = {},
                 **kwargs):
        """
        Initialization method.
        :param api_model: API model name.
        :param nickname_key: The field name to store the nickname
            relationship. It's "__dj__nickname__" in default.
        :param api_url: API URL. Defaults to DJ_API_URL environment variable.
        :param api_key: API key. Defaults to DJ_API_KEY environment variable.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for the calibration task.
        :param input_template: Template for building the model input.
        :param output_pattern: Regular expression for parsing model output.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param drop_text: If drop the text in the output.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.nickname_key = nickname_key

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.model_params = model_params
        self.sampling_params = sampling_params
        self.model_key = prepare_model(model_type='api',
                                       api_model=api_model,
                                       api_url=api_url,
                                       api_key=api_key,
                                       response_path=response_path,
                                       **model_params)

        self.try_num = try_num
        self.drop_text = drop_text

    def parse_output(self, raw_output):
        pattern = re.compile(self.output_pattern, re.VERBOSE | re.DOTALL)
        matches = pattern.findall(raw_output)

        nickname_relations = []

        for match in matches:
            _, role1, role2, role1_tmp, role2_tmp, nickname = match
            # for double check
            if role1.strip() != role1_tmp.strip() or role2.strip(
            ) != role2_tmp.strip():
                continue
            role1 = role1.strip()
            role2 = role2.strip()
            nickname = nickname.strip()
            # is name but not nickname
            if role2 == nickname:
                continue
            if role1 and role2 and nickname:
                nickname_relations.append((role1, role2, nickname))
            nickname_relations = list(set(nickname_relations))

        nickname_relations = [{
            'entity1': nr[0],
            'entity2': nr[1],
            'description': nr[2],
            'relation': 'nickname'
        } for nr in nickname_relations]

        return nickname_relations

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
        nickname_relations = []
        for i in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                nickname_relations = self.parse_output(output)
                if len(nickname_relations) > 0:
                    break
            except Exception as e:
                logger.warning(f'Exception: {e}')

        sample[self.nickname_key] = nickname_relations
        return sample
