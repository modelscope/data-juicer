import json
import logging
import re

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
    _batched_op = True

    def __init__(self,
                 hf_model: str = 'alibaba-pai/pai-qwen1_5-7b-doc2qa',
                 trust_remote_code=False,
                 pattern: str = None,
                 qa_format: str = 'chatml',
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.
        :param pattern: regular expression pattern to search for within text.
        :param qa_format: Output format of question and answer pair.
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

        if pattern is None:
            self.pattern = r'Human: (.*?)\nAssistant: (.*?)(?=\nHuman|$)'
        else:
            self.pattern = pattern

        self.qa_format = qa_format
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_model,
                                       trust_remote_code=trust_remote_code)

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

        inputs = processor(sample[self.text_key],
                           return_tensors='pt').to(model.device)
        response = model.generate(**inputs)
        output = processor.decode(response.cpu()[0], skip_special_tokens=True)
        qa_list = self._extract_qa(output)

        if not len(qa_list):
            logging.info(
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
