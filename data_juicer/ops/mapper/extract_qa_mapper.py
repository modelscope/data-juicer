import json
import re
from typing import Dict

from loguru import logger

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.model_utils import get_model, prepare_model


@OPERATORS.register_module('extract_qa_mapper')
class ExtractQAMapper(Mapper):
    """
    Mapper to extract question and answer pair from text samples.
    Recommended model list: [
        'alibaba-pai/pai-llama3-8b-doc2qa',
        'alibaba-pai/pai-baichuan2-7b-doc2qa',
        'alibaba-pai/pai-qwen1_5-4b-doc2qa',
        'alibaba-pai/pai-qwen1_5-7b-doc2qa',
        'alibaba-pai/pai-qwen1_5-1b8-doc2qa',
        'alibaba-pai/pai-qwen1_5-0b5-doc2qa'
    ]
    These recommended models are all trained with Chinese data
    and are suitable for Chinese.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 hf_model: str = 'alibaba-pai/pai-qwen1_5-7b-doc2qa',
                 pattern: str = None,
                 qa_format: str = 'chatml',
                 enable_vllm: bool = False,
                 tensor_parallel_size: int = None,
                 max_model_len: int = None,
                 max_num_seqs: int = 256,
                 sampling_params: Dict = {},
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.
        :param pattern: regular expression pattern to search for within text.
        :param qa_format: Output format of question and answer pair.
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

        The default data format parsed by this interface is as follows:
        Model Input:
            蒙古国的首都是乌兰巴托（Ulaanbaatar）
            冰岛的首都是雷克雅未克（Reykjavik）
        Model Output:
            蒙古国的首都是乌兰巴托（Ulaanbaatar）
            冰岛的首都是雷克雅未克（Reykjavik）
            Human: 请问蒙古国的首都是哪里？
            Assistant: 你好，根据提供的信息，蒙古国的首都是乌兰巴托（Ulaanbaatar）。
            Human: 冰岛的首都是哪里呢？
            Assistant: 冰岛的首都是雷克雅未克（Reykjavik）。
            ...
        """

        super().__init__(*args, **kwargs)
        self._accelerator = 'cuda'

        if pattern is None:
            self.pattern = r'Human: (.*?)\nAssistant: (.*?)(?=\nHuman|$)'
        else:
            self.pattern = pattern

        self.qa_format = qa_format
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
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs)
            self.sampling_params = SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_model)
            self.sampling_params = sampling_params

    def _extract_qa(self, output):
        """Extract qestion and answer pair from model output response."""
        qa_list = []

        pat = re.compile(self.pattern, re.DOTALL)
        qa_pairs = pat.findall(output)

        for _, qa in enumerate(qa_pairs, 1):
            user, assistant = qa
            qa_list.append((user.strip(), assistant.strip()))

        return qa_list

    def process(self, sample, rank=None):
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        if self.enable_vllm:
            response = model.generate([sample[self.text_key]],
                                      self.sampling_params)
            output = response[0].outputs[0].text
        else:
            inputs = processor(sample[self.text_key],
                               return_tensors='pt').to(model.device)
            response = model.generate(**inputs, **self.sampling_params)
            output = processor.decode(response.cpu()[0],
                                      skip_special_tokens=True)

        qa_list = self._extract_qa(output)

        if not len(qa_list):
            logger.info(
                'No question and answer data was extracted from this sample!')

        dialogue_data = []
        if self.qa_format == 'chatml':
            for qa in qa_list:
                dialogue_data.append({
                    'messages': [{
                        'role': 'user',
                        'content': qa[0]
                    }, {
                        'role': 'assistant',
                        'content': qa[1]
                    }]
                })
        else:
            raise ValueError(f'Not support {self.qa_format}!')

        sample[self.text_key] = json.dumps(dialogue_data, ensure_ascii=False)

        return sample
