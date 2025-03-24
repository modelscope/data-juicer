import json
import re
from typing import Dict, List, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (get_model, prepare_model,
                                           update_sampling_params)

torch = LazyLoader('torch', 'torch')
vllm = LazyLoader('vllm', 'vllm')

OP_NAME = 'llm_difficulty_score_filter'


@OPERATORS.register_module(OP_NAME)
class LLMDifficultyScoreFilter(Filter):
    """
    Filter to keep sample with high difficulty score estimated by LLM.
    """

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = """
You are an expert pedagogical evaluator for LLM training data. Analyze each data sample through multiple difficulty lenses and provide calibrated scores with detailed reasoning. Follow these guidelines:

1. Evaluation Dimensions
Rate each dimension (1-5 scale: 1=Novice-friendly, 3=Intermediate, 5=Expert-level):
- Linguistic Complexity: Vocabulary sophistication & syntactic structures
- Conceptual Depth: Abstraction level & theoretical requirements
- Prior Knowledge: Required domain-specific understanding
- Step Complexity: Problem-solving steps needed
- Ambiguity: Multiple valid interpretations

2. Output Format
json
{
  "dimension_scores": {
    "linguistic_complexity": ,
    "conceptual_depth": ,
    "prior_knowledge": ,
    "step_complexity": ,
    "ambiguity":
  },
  "flags": ["multistep_reasoning", "cultural_context", ...],
  "rationale": "Technical analysis of challenge sources"
}
3. Special Instructions
- Differentiate intrinsic vs. extrinsic difficulty factors
- Account for varying cultural/educational backgrounds
- Mark samples requiring cross-domain knowledge synthesis
- Consider temporal aspects for time-sensitive subjects
- Flag ambiguous samples needing difficulty bracketing
- Response a json dict

Example Response:

json
{
  "dimension_scores": {
    "linguistic_complexity": 3,
    "conceptual_depth": 5,
    "prior_knowledge": 4,
    "step_complexity": 4,
    "ambiguity": 5
  },
  "flags": ["nonlinear_reasoning", "semantic_ambiguity"],
  "rationale": "High conceptual difficulty due to multi-layered metaphor interpretation requiring philosophy background. Moderate linguistic complexity offset by implicit cultural references."
}
"""  # noqa: E501
    DEFAULT_INPUT_TEMPLATE = "# Data\n'''\n{data}\n'''\n\n# Response\njson\n"
    DEFAULT_FIELD_TEMPLATE = '**{field_name}**\n{field_data}'

    def __init__(self,
                 api_or_hf_model: str = 'gpt-4o',
                 min_score: float = 0.5,
                 *,
                 api_endpoint: Optional[str] = None,
                 response_path: Optional[str] = None,
                 input_keys: List[str] = ['text'],
                 field_names: List[str] = ['Text'],
                 system_prompt: Optional[str] = None,
                 input_template: Optional[str] = None,
                 field_template: Optional[str] = None,
                 try_num: PositiveInt = 3,
                 enable_vllm: bool = False,
                 model_params: Dict = {},
                 sampling_params: Dict = {},
                 **kwargs):
        """
        Initialization method.

        :param api_or_hf_model: API or huggingface model name.
        :param min_score: The lowest difficulty score threshold to keep
            the sample.
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
        :param enable_vllm: If true, use VLLM for loading hugging face or
            local llm. Otherwise, use API for reference.
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

        self.min_score = min_score
        self.try_num = try_num

        self.enable_vllm = enable_vllm

        sampling_params = update_sampling_params(sampling_params,
                                                 api_or_hf_model,
                                                 self.enable_vllm)

        if enable_vllm:
            assert torch.cuda.device_count() >= 1, 'must be executed in CUDA'
            # cannot initialize vllm replicas on different GPUs
            self.num_proc = 1
            if model_params.get('tensor_parallel_size') is None:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(f'Set tensor_parallel_size to \
                    {tensor_parallel_size} for vllm.')
                model_params['tensor_parallel_size'] = tensor_parallel_size
            self.model_key = prepare_model(
                model_type='vllm',
                pretrained_model_name_or_path=api_or_hf_model,
                **model_params)
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        else:
            self.sampling_params = sampling_params

            self.model_key = prepare_model(model_type='api',
                                           model=api_or_hf_model,
                                           endpoint=api_endpoint,
                                           response_path=response_path,
                                           **model_params)

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
        required_keys = [
            'linguistic_complexity', 'conceptual_depth', 'prior_knowledge',
            'step_complexity', 'ambiguity'
        ]

        total_score = 0
        for key in required_keys:
            total_score += dimension_scores[key]
        # div 5 for normalization
        avg_score = total_score / len(required_keys) / 5

        return avg_score, json_str

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.llm_difficulty_score in sample[Fields.stats]:
            return sample

        if self.enable_vllm:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

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
                if self.enable_vllm:
                    response = model.chat(messages, self.sampling_params)
                    output = response[0].outputs[0].text
                else:
                    output = model(messages, **self.sampling_params)
                score, record = self.parse_output(output)
                if record is not None:
                    break
            except Exception as e:
                logger.warning(f'Exception: {e}')

        sample[Fields.stats][StatsKeys.llm_difficulty_score] = score
        sample[Fields.stats][StatsKeys.llm_difficulty_record] = record

        return sample

    def process_single(self, sample, rank=None):
        itm_score = sample[Fields.stats][StatsKeys.llm_difficulty_score]

        return itm_score >= self.min_score
