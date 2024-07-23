from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model

DEFAULT_SYSTEM_PROMPT = '请优化这个指令，将其修改为一个更详细具体的指令。'


@OPERATORS.register_module('optimize_instruction_mapper')
class OptimizeInstructionMapper(Mapper):
    """Mapper to optimize instruction.
    Recommended model list: [
        alibaba-pai/Qwen2-1.5B-Instruct-Refine
        alibaba-pai/Qwen2-7B-Instruct-Refine
    ]
    """

    def __init__(self,
                 hf_model='alibaba-pai/Qwen2-7B-Instruct-Refine',
                 system_prompt=None,
                 enable_vllm=False,
                 tensor_parallel_size=None,
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.
        :param system_prompt: System prompt for optimize samples.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param tensor_parallel_size: It is only valid when enable_vllm is True.
            The number of GPUs to use for distributed execution with tensor
            parallelism.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        self.system_prompt = system_prompt
        self.enable_vllm = enable_vllm

        if enable_vllm:
            import torch
            from vllm import SamplingParams

            assert torch.cuda.device_count() >= 1, 'must be executed in CUDA'
            if not tensor_parallel_size:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(f'Set tensor_parallel_size to \
                    {tensor_parallel_size} for vllm.')
            self.model_key = prepare_model(
                model_type='vllm',
                pretrained_model_name_or_path=hf_model,
                tensor_parallel_size=tensor_parallel_size)
            self.sampling_params = SamplingParams(max_tokens=2048)
        else:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_model)

    def process(self, sample=None, rank=None):
        model, processor = get_model(self.model_key, rank=rank)

        messages = [{
            'role': 'system',
            'content': self.system_prompt
        }, {
            'role': 'user',
            'content': sample[self.text_key]
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
                                      eos_token_id=processor.eos_token_id)
            output = processor.decode(response.cpu()[0],
                                      skip_special_tokens=True)

        sample[self.text_key] = output

        return sample
