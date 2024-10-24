from typing import Dict, Optional

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE
from data_juicer.ops.mapper import OptimizeQAMapper
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader('torch', 'torch')
vllm = LazyLoader('vllm', 'vllm')

OP_NAME = 'optimize_query_mapper'


# TODO: Extend LLM-based OPs into API-based implementation.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class OptimizeQueryMapper(OptimizeQAMapper):
    """
    Mapper to optimize only query in question-answer pairs.
    """

    DEFAULT_SYSTEM_PROMPT = '优化问答对中的问题，将其更加详细具体，但仍可以由原答案回答。只输出优化后的问题，不要输出多余内容。'

    _accelerator = 'cuda'

    def __init__(self,
                 *,
                 hf_model: str = 'alibaba-pai/Qwen2-7B-Instruct-Refine',
                 trust_remote_code: bool = False,
                 system_prompt: Optional[str] = None,
                 enable_vllm: bool = True,
                 tensor_parallel_size: Optional[int] = None,
                 max_model_len: Optional[int] = None,
                 max_num_seqs: int = 256,
                 sampling_params: Dict = {},
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.
        :param trust_remote_code: passed to transformers
        :param system_prompt: System prompt for optimize samples.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param tensor_parallel_size: It is only valid when enable_vllm is True.
            The number of GPUs to use for distributed execution with tensor
            parallelism.
        :param max_model_len: It is only valid when enable_vllm is True.
            Model context length. If unspecified, will be automatically
            derived from the model config.
        :param max_num_seqs: It is only valid when enable_vllm is True.
            Maximum number of sequences to be processed in a single iteration.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(hf_model=hf_model,
                         trust_remote_code=trust_remote_code,
                         system_prompt=system_prompt,
                         enable_vllm=enable_vllm,
                         tensor_parallel_size=tensor_parallel_size,
                         max_model_len=max_model_len,
                         max_num_seqs=max_num_seqs,
                         sampling_params=sampling_params,
                         **kwargs)

    def build_input(self, sample):
        return sample[self.query_key]

    def parse_output(self, raw_output):
        return raw_output.strip(), None
