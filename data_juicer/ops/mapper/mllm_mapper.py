from typing import Dict

import torch

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.mm_utils import load_image
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'mllm_mapper'
torch.set_num_threads(1)


@LOADED_IMAGES.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class MllmMapper(Mapper):
    """Mapper to optimize instruction.
    Recommended model list: [
        liuhaotia/llava-v1.6-vicuna-7b
    ]
    """
    _accelerator = 'cuda'

    def __init__(self,
                 hf_model: str = 'liuhaotia/llava-v1.6-vicuna-7b',
                 max_new_tokens=256,
                 sampling_params: Dict = {},
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.
        :param sampling_params: Sampling hyperparameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, num_proc=1, **kwargs)

        self.hf_model = hf_model
        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_model)
        self.sampling_params = sampling_params
        self.max_new_tokens = max_new_tokens

    def process(self, sample=None, rank=None):

        model, processor = get_model(self.model_key, rank=rank, use_cuda=True)

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_key = sample[self.image_key]
        image = load_image(loaded_image_key)

        conversation = [
            {
                'role':
                'user',
                'content': [
                    {
                        'type': 'text',
                        'text': sample[self.text_key]
                    },
                    {
                        'type': 'image'
                    },
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation,
                                               add_generation_prompt=True)

        inputs = processor(images=image, text=prompt,
                           return_tensors='pt').to(model.device)

        response = model.generate(**inputs,
                                  max_new_tokens=self.max_new_tokens,
                                  **self.sampling_params)
        output = processor.decode(response.cpu()[0], skip_special_tokens=True)

        sample[self.text_key] = output

        return sample
