import json
import random
import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

from ..base_op import OPERATORS, Mapper

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")
rouge = LazyLoader("rouge")

OP_NAME = "generate_prompt_from_examples_mapper"


@OPERATORS.register_module(OP_NAME)
class GeneratePromptFromExamplesMapper(Mapper):
    """
    Mapper to generate prompts from examples.
    You should configure an empty dataset in your yaml config file:
    ```
    generated_dataset_config:
      type: 'EmptyFormatter'  # use `RayEmptyFormatter` when enable ray
      length: ${The number of generated samples}
      feature_keys: ${text key}
    ```
    The number of samples generated is determined by
    the length of the empty dataset.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "请你仔细观察多个示例提示词，按照你的理解，总结出相应规矩，然后写出一个新的更好的提示词，以让模型更好地完成指定任务。"
        "注意，新生成的【提示词】需要满足如下要求：\n"
        "1. 生成的【提示词】不能与输入的【提示词】完全一致，但是需要保持格式类似。\n"
        "2. 生成的【提示词】相比于输入的【提示词】不能有很大的变化，更多应该是关键词、核心参数等方面的微调。\n"
        "3. 【提示词】后可能会有一个0到1之间的评分用于表示人类对于该【提示词】在目标任务上的评分，如有的话请参考该评分生成可以获得更高评分的【提示词】；评分为-1表示该【提示词】的人类评分缺失，忽略该评分即可。\n"
        "4. 生成时只需生成【提示词】，不需生成其他任何额外信息（如【人类评分】等）。\n"
    )

    DEFAULT_INPUT_TEMPLATE = "{}"
    DEFAULT_EXAMPLE_TEMPLATE = "\n如下是一条示例数据：\n{}"
    DEFAULT_PROMPT_TEMPLATE = "【提示词】\n{}\n【人类评分】\n{}\n"
    DEFAULT_OUTPUT_PATTERN = r"【提示词】(.*?)(?=【人类评分】|$)"

    _batched_op = True
    _accelerator = "cuda"

    def __init__(
        self,
        api_or_hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
        seed_file: str = "",
        example_num: PositiveInt = 3,
        example_score_key: str = None,
        similarity_threshold: float = 0.8,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        example_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        enable_vllm: bool = False,
        is_hf_model: bool = False,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API or huggingface model name.
        :param seed_file: Path to the seed file in chatml format.
        :param example_num: The number of selected examples.
            Randomly select N examples from "seed_file" and
            put them into prompt as prompt examples.
        :param similarity_threshold: The similarity score threshold
            between the generated samples and the seed examples.
            Range from 0 to 1. Samples with similarity score less than
            this threshold will be kept.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for guiding the generation task.
        :param input_template: Template for building the input prompt. It must
            include one placeholder '{}', which will be replaced by
            `example_num` formatted examples defined by `example_template`.
        :param example_template: Template for formatting one prompt example. It
            must include one placeholder '{}', which will be replaced by one
            formatted prompt.
        :param prompt_template: Template for formatting a single prompt
            within each example. Must include two placeholders '{}' for the
            question and answer.
        :param output_pattern: Regular expression pattern to extract questions
            and answers from model response.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param is_hf_model:  If true, use Transformers for loading hugging face or
            local llm.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        if not seed_file:
            raise ValueError(
                "Please provide `seed_file` in chatml format."
                "Example: data-juicer/demos/data/demo-dataset-chatml.jsonl"
            )

        self.seed_file = seed_file
        self.example_num = example_num
        self.example_score_key = example_score_key
        self.similarity_threshold = similarity_threshold
        self.similarity_type = "rouge_l"

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.example_template = example_template or self.DEFAULT_EXAMPLE_TEMPLATE  # noqa: E501
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.enable_vllm = enable_vllm
        self.is_hf_model = is_hf_model
        model_params = model_params or {}
        sampling_params = sampling_params or {}

        sampling_params = update_sampling_params(sampling_params, api_or_hf_model, self.enable_vllm)

        if enable_vllm:
            assert torch.cuda.device_count() >= 1, "must be executed in CUDA"
            # cannot initialize vllm replicas on different GPUs
            self.num_proc = 1
            if model_params.get("tensor_parallel_size") is None:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(
                    f"Set tensor_parallel_size to \
                    {tensor_parallel_size} for vllm."
                )
                model_params["tensor_parallel_size"] = tensor_parallel_size
            self.model_key = prepare_model(
                model_type="vllm", pretrained_model_name_or_path=api_or_hf_model, **model_params
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        elif is_hf_model:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=api_or_hf_model,
                return_pipe=True,
                **model_params,
            )
            self.sampling_params = sampling_params
        else:
            self.sampling_params = sampling_params

            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )

        self.seed_prompt_samples = self._load_seed_samples()
        if len(self.seed_prompt_samples) == 0:
            raise ValueError("No prompt data was parsed from the seed file!")

    def _load_seed_samples(self):
        """Load prompts from jsonl format file."""
        prompt_samples = []
        with open(self.seed_file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                prompt, score = self._parse_prompt_str(line)
                if prompt:
                    prompt_samples.append((prompt, score))
        return prompt_samples

    def _max_rouge_l_score(self, hypothesis, references):
        r = rouge.Rouge()
        max_score = 0.0
        for reference in references:
            scores = r.get_scores(hypothesis, reference)
            rouge_l_score = scores[0]["rouge-l"]["f"]
            if rouge_l_score > max_score:
                max_score = rouge_l_score
        return max_score

    def _parse_prompt_str(self, sample_str):
        data = json.loads(sample_str)
        if self.prompt_key in data:
            score = data.get(self.example_score_key, -1)
            return data[self.prompt_key], score
        else:
            return None, None

    def build_input(self, prompt_examples):
        formatted_examples = "".join(
            [self.example_template.format(self.prompt_template.format(p, s)) for p, s in prompt_examples]
        )
        input_prompt = self.input_template.format(formatted_examples)
        return input_prompt

    def parse_output(self, raw_output):
        logger.debug(raw_output)
        output_prompt = ""
        matches = re.findall(self.output_pattern, raw_output, re.DOTALL)
        if len(matches) > 0:
            output_prompt = matches[0].strip()
        return output_prompt

    def process_single(self, sample, rank=None):
        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        random_prompt_samples = random.sample(self.seed_prompt_samples, self.example_num)
        input_prompt = self.build_input(random_prompt_samples)

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]

        if self.enable_vllm:
            response = model.chat(messages, self.sampling_params)
            output = response[0].outputs[0].text
        elif self.is_hf_model:
            # model is pipe
            response = model(messages, return_full_text=False, **self.sampling_params)
            output = response[0]["generated_text"]
        else:
            output = model(messages, **self.sampling_params)

        output_prompt = self.parse_output(output)
        if output_prompt == "":
            logger.warning("Parse model response error! " "No data generated for the current example!")
            sample.update({self.prompt_key: ""})
            return sample

        if self.similarity_type == "rouge_l":
            sim_score = self._max_rouge_l_score(output_prompt, [s[0] for s in random_prompt_samples])
        else:
            raise ValueError(f'Not support similarity type "{self.similarity_type}"!')

        if sim_score > self.similarity_threshold:
            output_prompt = ""
            logger.info("Filter this generated sample due to similarity.")

        sample.update({self.prompt_key: output_prompt})
        return sample
