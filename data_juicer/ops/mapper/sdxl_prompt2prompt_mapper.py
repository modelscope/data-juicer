import logging

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.ops.op_fusion import LOADED_IMAGES
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

diffusers = LazyLoader('diffusers', 'diffusers')
torch = LazyLoader('torch', 'torch')
p2p_pipeline = LazyLoader('p2p_pipeline',
                          'data_juicer.ops.common.prompt2prompt_pipeline',
                          auto_install=False)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OP_NAME = 'sdxl_prompt2prompt_mapper'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class SDXLPrompt2PromptMapper(Mapper):
    """
        Generate pairs of similar images by the SDXL model
    """

    _accelerator = 'cuda'

    def __init__(
            self,
            hf_diffusion: str = 'stabilityai/stable-diffusion-xl-base-1.0',
            trust_remote_code=False,
            torch_dtype: str = 'fp32',
            num_inference_steps: float = 50,
            guidance_scale: float = 7.5,
            text_key_second=None,
            text_key_third=None,
            *args,
            **kwargs):
        """
        Initialization method.

        :param hf_diffusion: diffusion model name on huggingface to generate
            the image.
        :param torch_dtype: the floating point type used to load the diffusion
            model.
        :param num_inference_steps: The larger the value, the better the
        image generation quality; however, this also increases the time
        required for generation.
        :param guidance_scale: A higher guidance scale value encourages the
            model to generate images closely linked to the text prompt at the
            expense of lower image quality. Guidance scale is enabled when
        :param text_key_second: used to store the first caption
            in the caption pair.
        :param text_key_third: used to store the second caption
            in the caption pair.

        """
        kwargs.setdefault('mem_required', '38GB')
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.hf_diffusion = hf_diffusion
        self.torch_dtype = torch_dtype
        self.model_key = prepare_model(
            model_type='sdxl-prompt-to-prompt',
            pretrained_model_name_or_path=hf_diffusion,
            pipe_func=p2p_pipeline.Prompt2PromptPipeline,
            torch_dtype=torch_dtype)
        self.text_key_second = text_key_second
        self.text_key_third = text_key_third

    def process_single(self, sample, rank=None, context=False):

        if self.text_key_second is None:
            logger.error('This OP (sdxl_prompt2prompt_mapper) requires \
                processing multiple fields, and you need to specify \
                valid `text_key_second`')

        if self.text_key_third is None:
            logger.error('This OP (sdxl_prompt2prompt_mapper) requires \
                processing multiple fields, and you need to specify \
                valid `text_key_third`')

        model = get_model(model_key=self.model_key,
                          rank=rank,
                          use_cuda=self.use_cuda())

        seed = 0
        g_cpu = torch.Generator().manual_seed(seed)

        cross_attention_kwargs = {
            'edit_type': 'refine',
            'n_self_replace': 0.4,
            'n_cross_replace': {
                'default_': 1.0,
                'confetti': 0.8
            },
        }

        sample[self.image_key] = []

        with torch.no_grad():
            prompts = [
                sample[self.text_key_second], sample[self.text_key_third]
            ]
            image = model(prompts,
                          cross_attention_kwargs=cross_attention_kwargs,
                          guidance_scale=self.guidance_scale,
                          num_inference_steps=self.num_inference_steps,
                          generator=g_cpu)

            for idx, img in enumerate(image[self.image_key]):
                sample[self.image_key].append(img)

        return sample
