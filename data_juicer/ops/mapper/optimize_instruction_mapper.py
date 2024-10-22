from typing import Dict, Optional

from loguru import logger

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE, Mapper
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

torch = LazyLoader('torch', 'torch')
vllm = LazyLoader('vllm', 'vllm')

DEFAULT_SYSTEM_PROMPT = '请优化这个指令，将其修改为一个更详细具体的指令。'

OP_NAME = 'optimize_instruction_mapper'


# TODO: Extend LLM-based OPs into API-based implementation.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class OptimizeInstructionMapper(Mapper):
    """Mapper to optimize instruction.
    Recommended model list: [
        alibaba-pai/Qwen2-1.5B-Instruct-Refine
        alibaba-pai/Qwen2-7B-Instruct-Refine
    ]
    """
    _accelerator = 'cuda'

    def __init__(self,
                 hf_model: str = 'alibaba-pai/Qwen2-7B-Instruct-Refine',
                 trust_remote_code: bool = False,
                 system_prompt: Optional[str] = None,
                 enable_vllm: bool = True,
                 tensor_parallel_size: Optional[int] = None,
                 max_model_len: Optional[int] = None,
                 max_num_seqs: int = 256,
                 sampling_params: Dict = {},
                 *args,
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
        super().__init__(*args, **kwargs)
        self.num_proc = 1

        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt = system_prompt
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

    def process_single(self, sample=None, rank=None):
        model, processor = get_model(self.model_key, rank=rank)

        messages = [{
            'role': 'system',
            'content': self.system_prompt
        }, {
            'role': 'user',
            'content': sample[self.query_key]
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

        sample[self.query_key] = output

        return sample
