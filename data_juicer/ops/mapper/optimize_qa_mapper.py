import re
from typing import Dict, Optional

from loguru import logger

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE, Mapper
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

torch = LazyLoader('torch', 'torch')
vllm = LazyLoader('vllm', 'vllm')

OP_NAME = 'optimize_qa_mapper'


# TODO: Extend LLM-based OPs into API-based implementation.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class OptimizeQAMapper(Mapper):
    """
    Mapper to optimize question-answer pairs.
    """

    # avoid leading whitespace
    DEFAULT_SYSTEM_PROMPT = ('请优化输入的问答对，使【问题】和【回答】都更加详细、准确。\n'
                             '按照以下格式输出：\n'
                             '【问题】\n'
                             '优化后的问题\n'
                             '【回答】\n'
                             '优化后的回答')
    DEFAULT_INPUT_TEMPLATE = '以下是原始问答对：\n\n{qa_pair}'
    DEFAULT_QA_PAIR_TEMPLATE = '【问题】\n{}\n【回答】\n{}\n'
    DEFAULT_OUTPUT_PATTERN = r'【问题】\s*(.*?)\s*【回答】\s*(.*)'

    _accelerator = 'cuda'

    def __init__(self,
                 *,
                 hf_model: str = 'Qwen/Qwen-7B-Chat',
                 trust_remote_code: bool = False,
                 system_prompt: Optional[str] = None,
                 input_template: Optional[str] = None,
                 qa_pair_template: Optional[str] = None,
                 output_pattern: Optional[str] = None,
                 enable_vllm: bool = True,
                 tensor_parallel_size: Optional[int] = None,
                 max_model_len: Optional[int] = None,
                 max_num_seqs: int = 256,
                 sampling_params: Dict = {},
                 **kwargs):
        """
        Initialization method.

        :param hf_model: Hugging Face model ID.
        :param trust_remote_code: Whether to trust remote code from the model
            (passed to transformers).
        :param system_prompt: System prompt for the optimization task.
        :param input_template: Template for building the input for the model.
        :param qa_pair_template: Template for formatting the question-answer
            pair.
        :param output_pattern: Pattern for parsing the output from the model.
        :param enable_vllm: Whether to use VLLM for inference acceleration.
        :param tensor_parallel_size: Number of GPUs for distributed execution,
            valid only if VLLM is enabled.
        :param max_model_len: Model context length, valid only if VLLM is
            enabled. If unspecified, will be derived from the model config.
        :param max_num_seqs: Max number of sequences to process at once, valid
            only if VLLM is enabled.
        :param sampling_params: Sampling parameters for text generation (e.g.,
            {'temperature': 0.9, 'top_p': 0.95}).
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_proc = 1

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN
        self.qa_pair_template = qa_pair_template or \
            self.DEFAULT_QA_PAIR_TEMPLATE
        self.enable_vllm = enable_vllm

        if enable_vllm:
            assert torch.cuda.device_count() >= 1, 'must be executed in CUDA'
            if not tensor_parallel_size:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(f'Set tensor_parallel_size to \
                    {tensor_parallel_size} for vllm.')
            self.model_key = prepare_model(
                model_type='vllm',
                pretrained_model_name_or_path=hf_model,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs)
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_model,
                trust_remote_code=trust_remote_code)
            self.sampling_params = sampling_params

    def build_input(self, sample):
        qa_pair = self.qa_pair_pattern.format(sample[self.query_key],
                                              sample[self.response_key])
        input_prompt = self.input_pattern.format(qa_pair)
        return input_prompt

    def parse_output(self, raw_output):
        match = re.match(self.output_pattern, raw_output)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        else:
            return None, None

    def process_single(self, sample=None, rank=None):
        model, processor = get_model(self.model_key, rank=rank)

        messages = [{
            'role': 'system',
            'content': self.system_prompt
        }, {
            'role': 'user',
            'content': self.build_input(sample)
        }]
        input_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

        if self.enable_vllm:
            response = model.generate([input_prompt], self.sampling_params)
            output = response[0].outputs[0].text
        else:
            inputs = processor(input_prompt,
                               return_tensors='pt').to(model.device)
            response = model.generate(**inputs,
                                      eos_token_id=processor.eos_token_id,
                                      **self.sampling_params)
            output = processor.decode(response.cpu()[0],
                                      skip_special_tokens=True)

        parsed_q, parsed_a = self.parse_output(output)
        if parsed_q:
            sample[self.query_key] = parsed_q
        if parsed_a:
            sample[self.response_key] = parsed_a

        return sample
