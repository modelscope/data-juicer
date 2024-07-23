# Some code here has been modified from:
# https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1/
# --------------------------------------------------------
import json
import random
import re

from loguru import logger

from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

DEFAULT_PROMPT_TEMPLATE = """
请你仔细观察多个示例数据的输入和输出，按照你的理解，总结出相应规矩，然后写出一个新的【问题】和【回答】。注意，新生成的【问题】和【回答】需要满足如下要求：
1. 生成的【问题】和【回答】不能与输入的【问题】和【回答】一致，但是需要保持格式相同。
2. 生成的【问题】不一定要局限于输入【问题】的话题或领域，生成的【回答】需要正确回答生成的【问题】。
3. 提供的【问题】和【回答】可能是多轮对话，生成的【问题】和【回答】也可以是多轮，但是需要保持格式相同。
4. 生成的【问题】和【回答】必须成对出现，而且【问题】需要在【回答】之前。
{augmented_data}
"""
QA_EXTRACTION_PATTERN = r'【问题】\s*(.*?)\s*【回答】\s*(.*?)\s*(?=【问题】|$)'
SAMPLE_INTRODUCTION_AND_QA_FORMAT = ('\n如下是一条示例数据：\n\n'
                                     '{qa_pairs}')
QA_PAIR_FORMAT = '【问题】\n{}\n【回答】\n{}\n'


@OPERATORS.register_module('generate_instruction_mapper')
class GenerateInstructionMapper(Mapper):
    """Mapper to generate new instruction text data.
    You should configure an empty dataset in your yaml config file:
    ```
    dataset_config:
      type: 'EmptyFormatter'
      length: ${The number of generated samples}
    ```
    The number of samples generated is determined by
    the length of the empty dataset.
    """

    def __init__(self,
                 hf_model,
                 seed_file,
                 instruct_num,
                 similarity_threshold=0.7,
                 prompt_template=None,
                 enable_vllm=False,
                 tensor_parallel_size=None,
                 sampling_params={},
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param hf_model: Hugginface model id.
        :param seed_file: Seed file path, chatml format.
        :param instruct_num: The number of instruction samples.
            Randomly select N samples from "seed_file" and
            put them into prompt as instruction samples.
        :param generate_num: The number of generated samples.
        :param similarity_threshold: The similarity score threshold
            between the generated samples and the seed samples.
            Range from 0 to 1. Samples with similarity score less than
            this threshold will be kept.
        :param prompt_template: Prompt template for generate samples.
            Please make sure the template contains "{augmented_data}",
            which corresponds to the augmented samples.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param tensor_parallel_size: It is only valid when enable_vllm is True.
            The number of GPUs to use for distributed execution with tensor
            parallelism.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)

        self.instruct_num = instruct_num
        self.similarity_threshold = similarity_threshold
        self.similarity_type = 'rouge_l'
        # self._batched_op = True

        if prompt_template is None:
            prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.prompt_template = prompt_template
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
            self.sampling_params = SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_model)
            self.sampling_params = sampling_params

        self.seed_qa_samples = self.load_seed_qa_samples(seed_file)

        if len(self.seed_qa_samples) == 0:
            raise ValueError('No QA data was parsed from the seed file!')

        self.reference_samples = [
            '\n'.join(['\n'.join(qa_pair) for qa_pair in qa_pairs]) + '\n'
            for qa_pairs in self.seed_qa_samples
        ]

    def load_seed_qa_samples(self, seed_file):
        """Load QA pairs from chatml format file."""
        qa_samples = []
        with open(seed_file) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                qa_pairs = self.parse_chatml_str(line)
                if len(qa_pairs) > 0:
                    qa_samples.append(qa_pairs)

        return qa_samples

    def build_prompt(self, qa_samples, prompt_template):

        def format_qa_pairs(qa_pairs):
            return ''.join(
                [QA_PAIR_FORMAT.format(q, a) for q, a in qa_pairs if q and a])

        body_fragments = [
            SAMPLE_INTRODUCTION_AND_QA_FORMAT.format(
                qa_pairs=format_qa_pairs(qa_pairs)) for qa_pairs in qa_samples
        ]

        body = ''.join(body_fragments)

        return prompt_template.format(augmented_data=body)

    def parse_chatml_str(self, input_str):
        user_input = None
        assistant_output = None
        qa_pairs = []
        data = json.loads(input_str)
        for message in data['messages']:
            role = message['role']
            content = message['content']
            if role == 'user':
                user_input = content
            elif role == 'assistant':
                assistant_output = content
                qa_pairs.append((user_input, assistant_output))
        return qa_pairs

    def parse_response(self, response_str):
        pattern = QA_EXTRACTION_PATTERN
        matches = re.findall(pattern, response_str, re.DOTALL)
        response_str = ''
        out_qa_pairs = []
        for i, match in enumerate(matches):
            question, answer = match
            question = question.strip()
            answer = answer.strip()
            out_qa_pairs.append((question, answer))
            response_str += question + '\n' + answer + '\n'

        if len(out_qa_pairs) == 0:
            logger.error('Parse model response error! '
                         'No data generated for the current response!')

        return out_qa_pairs, response_str

    def max_rouge_l_score(self, reference, candidates):
        from rouge import Rouge

        rouge = Rouge()
        max_score = 0.0
        for candidate in candidates:
            scores = rouge.get_scores(candidate, reference)
            rouge_l_score = scores[0]['rouge-l']['f']
            if rouge_l_score > max_score:
                max_score = rouge_l_score
        return max_score

    def process(self, sample=None, rank=None):
        model, processor = get_model(self.model_key, rank=rank)

        random_qa_samples = random.sample(self.seed_qa_samples,
                                          self.instruct_num)
        input_prompt = self.build_prompt(random_qa_samples,
                                         self.prompt_template)
        if self.enable_vllm:
            response = model.generate([input_prompt], self.sampling_params)
            response_str = response[0].outputs[0].text
        else:
            inputs = processor(input_prompt,
                               return_tensors='pt').to(model.device)
            output_ids = model.generate(**inputs, **self.sampling_params)
            # remove the input prompt from the output
            output_ids = output_ids[:, inputs.data['input_ids'].shape[1]:]
            response_str = processor.decode(output_ids.cpu()[0],
                                            skip_special_tokens=True)
        message_list = []
        out_qa_pairs, response_str = self.parse_response(response_str)

        if not response_str:
            return {self.text_key: json.dumps({'messages': message_list})}

        if self.similarity_type == 'rouge_l':
            sim_score = self.max_rouge_l_score(response_str,
                                               self.reference_samples)
        else:
            raise ValueError(
                f'Not support similarity type "{self.similarity_type}"!')

        if sim_score <= self.similarity_threshold:
            for question, answer in out_qa_pairs:
                message_list.append({'role': 'user', 'content': question})
                message_list.append({'role': 'assistant', 'content': answer})
        else:
            logger.info('Filter this generated sample due to similarity.')

        return {
            self.text_key:
            json.dumps({'messages': message_list}, ensure_ascii=False)
        }
