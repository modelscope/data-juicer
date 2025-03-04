import json
import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'llm_api_quality_score_filter'


@OPERATORS.register_module(OP_NAME)
class LLMAPIQualityScoreFilter(Filter):
    """
    Filter to keep sample with high quality score estimated by LLM in API.
    """

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = """
You are a meticulous data quality assessor for LLM training. Analyze each data sample across multiple quality dimensions and provide numerical scores with reasoning. Follow these guidelines:

1. Evaluation Dimensions
Score each dimension (1-5 scale: 1=lowest, 5=highest):
- Accuracy: Factual correctness & verifiability
- Grammar: Linguistic correctness & fluency
- Informativeness: Depth/utility of content
- Coherence: Logical structure & consistency

2. Scoring Protocol
- Base scores on concrete evidence from text
- Flag samples needing human review (confidence <90%)
- Compare with similar data points for consistency
- Penalize hallucination/misinformation severely

3. Output Format
json
{
  "dimension_scores": {
    "accuracy": ,
    "grammar": ,
    "informativeness": ,
    "coherence": ,
  },
  "flags": ["syntax_error", "insufficient_information", ...],
  "rationale": "Concise technical analysis",
  "recommendation": ["keep", "review", "discard"]
}
4. Special Instructions
- Prioritize factual integrity over stylistic qualities
- Treat unverified medical/legal claims as high-risk
- Contextualize cultural references appropriately
- Response a json dict

Example Response:

json
{
  "dimension_scores": {
    "accuracy": 2,
    "grammar": 4,
    "informativeness": 4,
    "coherence": 2,
  },
  "flags": ["accuracy_concern", "logical_confusion"],
  "rationale": "The text provides rich information but suffers from logical confusion and lacks contextual coherence. Excellent grammatical structure offset by factual inaccuracies.",
  "recommendation": "review"
}
"""  # noqa: E501
    DEFAULT_INPUT_TEMPLATE = "# Data\n'''\n{data}\n'''\n\n# Response\njson\n"
    DEFAULT_FIELD_TEMPLATE = '**{field_name}**\n{field_data}'

    def __init__(self,
                 api_model: str = 'gpt-4o',
                 score_threshold: float = 0.5,
                 *,
                 api_endpoint: Optional[str] = None,
                 response_path: Optional[str] = None,
                 input_keys: List[str] = ['text'],
                 field_names: List[str] = ['Text'],
                 system_prompt: Optional[str] = None,
                 input_template: Optional[str] = None,
                 field_template: Optional[str] = None,
                 try_num: PositiveInt = 3,
                 model_params: Dict = {},
                 sampling_params: Dict = {},
                 **kwargs):
        """
        Initialization method.

        :param api_model: API model name.
        :param score_threshold: The lowest quality score threshold to keep the
            sample.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param input_keys: Sub set of keys in the sample. Support data with
            multi fields such as 'query', 'analysis' and 'answer' in RFT data.
        :param field_names: Corresponding field names for input keys.
        :param system_prompt: System prompt for the task.
        :param input_template: Template for building the model input.
        :param field_template: Template for each field in the prompt.
        :param try_num: The number of retry attempts when there is an API
            call error or output parsing error.
        :param model_params: Parameters for initializing the API model.
        :param sampling_params: Extra parameters passed to the API call.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        assert len(input_keys) == len(
            field_names
        ), 'The input_keys and field_names must correspond one-to-one!'
        self.input_keys = input_keys
        self.field_names = field_names
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.field_template = field_template or self.DEFAULT_FIELD_TEMPLATE

        self.score_threshold = score_threshold

        self.sampling_params = sampling_params

        self.model_key = prepare_model(model_type='api',
                                       model=api_model,
                                       endpoint=api_endpoint,
                                       response_path=response_path,
                                       **model_params)

        self.try_num = try_num

    def build_input(self, sample):
        if not set(self.input_keys) <= set(sample.keys()):
            logger.warning(
                f'Not all input keys {self.input_keys} are in the sample!')
        field_strs = [
            self.field_template.format(field_name=n, field_data=sample[k])
            for (k, n) in zip(self.input_keys, self.field_names) if k in sample
        ]
        data_str = '\n\n'.join(field_strs)
        input_prompt = self.input_template.format(data=data_str)

        return input_prompt

    def parse_output(self, raw_output):

        def extract_outer_braces(s):
            pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            match = re.search(pattern, s)

            if match:
                return match.group(1)
            else:
                return None

        json_str = extract_outer_braces(raw_output)
        data = json.loads(json_str)

        dimension_scores = data['dimension_scores']
        required_keys = ['accuracy', 'grammar', 'informativeness', 'coherence']

        total_score = 0
        for key in required_keys:
            total_score += dimension_scores[key]
        # div 5 for normalization
        avg_score = total_score / len(required_keys) / 5

        return avg_score, json_str

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.llm_quality_score in sample[Fields.stats]:
            return sample

        client = get_model(self.model_key, rank=rank)

        messages = [{
            'role': 'system',
            'content': self.system_prompt
        }, {
            'role': 'user',
            'content': self.build_input(sample)
        }]
        score, record = 0, None
        for _ in range(self.try_num):
            try:
                output = client(messages, **self.sampling_params)
                score, record = self.parse_output(output)
                if record is not None:
                    break
            except Exception as e:
                logger.warning(f'Exception: {e}')

        sample[Fields.stats][StatsKeys.llm_quality_score] = score
        sample[Fields.stats][StatsKeys.llm_quality_record] = record

        return sample

    def process_single(self, sample, rank=None):
        itm_score = sample[Fields.stats][StatsKeys.llm_quality_score]

        return itm_score >= self.score_threshold
