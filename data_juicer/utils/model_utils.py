import fnmatch
import inspect
import os
from functools import partial
from pickle import UnpicklingError
from typing import Optional, Union

import httpx
import multiprocess as mp
import wget
from loguru import logger

from data_juicer import cuda_device_count
from data_juicer.utils.lazy_loader import AUTOINSTALL, LazyLoader

from .cache_utils import DATA_JUICER_MODELS_CACHE as DJMC

torch = LazyLoader('torch', 'torch')
transformers = LazyLoader('transformers', 'transformers')
nn = LazyLoader('nn', 'torch.nn')
fasttext = LazyLoader('fasttext', 'fasttext')
sentencepiece = LazyLoader('sentencepiece', 'sentencepiece')
kenlm = LazyLoader('kenlm', 'kenlm')
nltk = LazyLoader('nltk', 'nltk')
aes_pre = LazyLoader('aes_pre', 'aesthetics_predictor')
vllm = LazyLoader('vllm', 'vllm')
diffusers = LazyLoader('diffusers', 'diffusers')
ram = LazyLoader('ram', 'ram.models')
cv2 = LazyLoader('cv2', 'cv2')
openai = LazyLoader('openai', 'openai')

MODEL_ZOO = {}

# Default cached models links for downloading
MODEL_LINKS = 'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/' \
               'data_juicer/models/'

# Backup cached models links for downloading
BACKUP_MODEL_LINKS = {
    # language identification model from fasttext
    'lid.176.bin':
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/',

    # tokenizer and language model for English from sentencepiece and KenLM
    '*.sp.model':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',
    '*.arpa.bin':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',

    # sentence split model from nltk punkt
    'punkt.*.pickle':
    'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/'
    'data_juicer/models/',
}


def get_backup_model_link(model_name):
    for pattern, url in BACKUP_MODEL_LINKS.items():
        if fnmatch.fnmatch(model_name, pattern):
            return url
    return None


def check_model(model_name, force=False):
    """
    Check whether a model exists in DATA_JUICER_MODELS_CACHE.
    If exists, return its full path.
    Else, download it from cached models links.

    :param model_name: a specified model name
    :param force: Whether to download model forcefully or not, Sometimes
        the model file maybe incomplete for some reason, so need to
        download again forcefully.
    """
    # check for local model
    if os.path.exists(model_name):
        return model_name

    if not os.path.exists(DJMC):
        os.makedirs(DJMC)

    # check if the specified model exists. If it does not exist, download it
    cached_model_path = os.path.join(DJMC, model_name)
    if force:
        if os.path.exists(cached_model_path):
            os.remove(cached_model_path)
            logger.info(
                f'Model [{cached_model_path}] is invalid. Forcing download...')
        else:
            logger.info(
                f'Model [{cached_model_path}] is not found. Downloading...')

        try:
            model_link = os.path.join(MODEL_LINKS, model_name)
            wget.download(model_link, cached_model_path)
        except:  # noqa: E722
            try:
                backup_model_link = os.path.join(
                    get_backup_model_link(model_name), model_name)
                wget.download(backup_model_link, cached_model_path)
            except:  # noqa: E722
                logger.error(
                    f'Downloading model [{model_name}] error. '
                    f'Please retry later or download it into {DJMC} '
                    f'manually from {model_link} or {backup_model_link} ')
                exit(1)
    return cached_model_path


class APIModel:

    def __init__(self, model, endpoint=None, response_path=None, **kwargs):
        """
        Initializes an instance of the APIModel class.

        :param model: The name of the model to be used for making API
            calls. This should correspond to a valid model identifier
            recognized by the API server.
        :param endpoint: The URL endpoint for the API. If provided as a
            relative path, it will be appended to the base URL (defined by the
            `OPENAI_BASE_URL` environment variable or through an additional
            `base_url` parameter). Defaults to '/chat/completions' for
            OpenAI compatibility.
        :param response_path: A dot-separated string specifying the path to
            extract the desired content from the API response. The default
            value is 'choices.0.message.content', which corresponds to the
            typical structure of an OpenAI API response.
        :param kwargs: Additional keyword arguments for configuring the
            internal OpenAI client.
        """
        self.model = model
        self.endpoint = endpoint or '/chat/completions'
        self.response_path = response_path or 'choices.0.message.content'

        client_args = self._filter_arguments(openai.OpenAI, kwargs)
        self._client = openai.OpenAI(**client_args)

    def __call__(self, messages, **kwargs):
        """
        Sends messages to the configured API model and returns the parsed
        response content.

        :param messages: A list of message dictionaries to send to the API.
                         Each message should have a 'role' (e.g., 'user',
                         'assistant') and 'content' (the message text).
        :param kwargs: Additional parameters for the API call.
        :return: The parsed response content from the API call, or an empty
            string if an error occurs.
        """
        body = {
            'messages': messages,
            'model': self.model,
        }
        body.update(kwargs)
        stream = kwargs.get('stream', False)
        stream_cls = openai.Stream[openai.types.chat.ChatCompletionChunk]

        try:
            response = self._client.post(self.endpoint,
                                         body=body,
                                         cast_to=httpx.Response,
                                         stream=stream,
                                         stream_cls=stream_cls)
            result = response.json()
            return self._nested_access(result, self.response_path)
        except Exception as e:
            logger.exception(e)
            return ''

    @staticmethod
    def _nested_access(data, path):
        """
        Access nested data using a dot-separated path.

        :param data: A dictionary or a list to access the nested data from.
        :param path: A dot-separated string representing the path to access.
                     This can include numeric indices when accessing list
                     elements.
        :return: The value located at the specified path, or raises a KeyError
                 or IndexError if the path does not exist.
        """
        keys = path.split('.')
        for key in keys:
            # Convert string keys to integers if they are numeric
            key = int(key) if key.isdigit() else key
            data = data[key]
        return data

    @staticmethod
    def _filter_arguments(func, args_dict):
        """
        Filters and returns only the valid arguments for a given function
        signature.

        :param func: The function or callable to inspect.
        :param args_dict: A dictionary of argument names and values to filter.
        :return: A dictionary containing only the arguments that match the
                 function's signature, preserving any **kwargs if applicable.
        """
        params = inspect.signature(func).parameters
        filtered_args = {}
        for name, param in params.items():
            # If **kwargs is found, return without change
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return args_dict
            # Collect valid parameters
            if name not in {'self', 'cls'} and name in args_dict:
                filtered_args[name] = args_dict[name]
        return filtered_args


def prepare_api_model(model,
                      *,
                      endpoint=None,
                      response_path=None,
                      return_processor=False,
                      processor_config=None,
                      **model_params):
    """
    Creates an instance of the APIModel for interacting with OpenAI-like APIs.

    :param model: The name of the model to be used for making API calls.
    :param endpoint: The URL endpoint for the API. If provided as a relative
        path, it will be appended to the base URL (defined by the
        `OPENAI_BASE_URL` environment variable or through an additional
        `base_url` parameter). By default, it is set to
        '/chat/completions' for OpenAI compatibility.
    :param response_path: A dot-separated string specifying the path to
        extract desired content from the API response. The default value is
        'choices.0.message.content', which corresponds to the typical
        structure of an OpenAI API response.
    :param return_processor: A boolean flag indicating whether to return a
        processor along with the model. The processor can be used for tasks
        like tokenization or encoding. Defaults to False.
    :param processor_config: A dictionary containing configuration parameters
        for initializing a Hugging Face processor. It is only relevant if
        `return_processor` is set to True.
    :param model_params: Additional parameters for configuring the API model.
    :return: A callable APIModel instance, and optionally a processor
        if `return_processor` is True.
    """
    client = APIModel(model=model,
                      endpoint=endpoint,
                      response_path=response_path,
                      **model_params)

    if not return_processor:
        return client

    def get_processor():
        try:
            import tiktoken
            return tiktoken.encoding_for_model(model)
        except Exception:
            pass

        try:
            import dashscope
            return dashscope.get_tokenizer(model)
        except Exception:
            pass

        try:
            processor = transformers.AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model, **processor_config)
            return processor
        except Exception:
            pass

        raise ValueError(
            'Failed to initialize the processor. Please check the following:\n'  # noqa: E501
            "- For OpenAI models: Install 'tiktoken' via `pip install tiktoken`.\n"  # noqa: E501
            "- For DashScope models: Install both 'dashscope' and 'tiktoken' via `pip install dashscope tiktoken`.\n"  # noqa: E501
            "- For custom models: Use the 'processor_config' parameter to configure a Hugging Face processor."  # noqa: E501
        )

    if processor_config is not None \
            and 'pretrained_model_name_or_path' in processor_config:
        processor = transformers.AutoProcessor.from_pretrained(
            **processor_config)
    else:
        processor = get_processor()
    return (client, processor)


def prepare_diffusion_model(pretrained_model_name_or_path, diffusion_type,
                            **model_params):
    """
        Prepare and load an Diffusion model from HuggingFace.

        :param pretrained_model_name_or_path: input Diffusion model name
            or local path to the model
        :param diffusion_type: the use of the diffusion model. It can be
            'image2image', 'text2image', 'inpainting'
        :return: a Diffusion model.
    """
    AUTOINSTALL.check(['torch', 'transformers'])

    if 'device' in model_params:
        model_params['device_map'] = model_params.pop('device')

    diffusion_type_to_pipeline = {
        'image2image': diffusers.AutoPipelineForImage2Image,
        'text2image': diffusers.AutoPipelineForText2Image,
        'inpainting': diffusers.AutoPipelineForInpainting
    }

    if diffusion_type not in diffusion_type_to_pipeline.keys():
        raise ValueError(
            f'Not support {diffusion_type} diffusion_type for diffusion '
            'model. Can only be one of '
            '["image2image", "text2image", "inpainting"].')

    pipeline = diffusion_type_to_pipeline[diffusion_type]
    model = pipeline.from_pretrained(pretrained_model_name_or_path,
                                     **model_params)

    return model


def prepare_fasttext_model(model_name='lid.176.bin', **model_params):
    """
    Prepare and load a fasttext model.

    :param model_name: input model name
    :return: model instance.
    """
    logger.info('Loading fasttext language identification model...')
    try:
        ft_model = fasttext.load_model(check_model(model_name))
    except:  # noqa: E722
        ft_model = fasttext.load_model(check_model(model_name, force=True))
    return ft_model


def prepare_huggingface_model(pretrained_model_name_or_path,
                              *,
                              return_model=True,
                              return_pipe=False,
                              pipe_task='text-generation',
                              **model_params):
    """
    Prepare and load a HuggingFace model with the correspoding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param return_pipe: whether to wrap model into pipeline
    :param model_params: model initialization parameters.
    :return: a tuple of (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    # require torch for transformer model
    AUTOINSTALL.check(['torch'])

    if 'device' in model_params:
        model_params['device_map'] = model_params.pop('device')

    processor = transformers.AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, **model_params)

    if return_model:
        config = transformers.AutoConfig.from_pretrained(
            pretrained_model_name_or_path, **model_params)
        if hasattr(config, 'auto_map'):
            class_name = next(
                (k for k in config.auto_map if k.startswith('AutoModel')),
                'AutoModel')
        else:
            # TODO: What happens if more than one
            class_name = config.architectures[0]

        model_class = getattr(transformers, class_name)
        model = model_class.from_pretrained(pretrained_model_name_or_path,
                                            **model_params)

        if return_pipe:
            if isinstance(processor, transformers.PreTrainedTokenizerBase):
                pipe_params = {'tokenizer': processor}
            elif isinstance(processor, transformers.SequenceFeatureExtractor):
                pipe_params = {'feature_extractor': processor}
            elif isinstance(processor, transformers.BaseImageProcessor):
                pipe_params = {'image_processor': processor}
            pipe = transformers.pipeline(task=pipe_task,
                                         model=model,
                                         config=config,
                                         **pipe_params)
            model = pipe

    return (model, processor) if return_model else processor


def prepare_kenlm_model(lang, name_pattern='{}.arpa.bin', **model_params):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    model_params.pop('device', None)

    model_name = name_pattern.format(lang)

    logger.info('Loading kenlm language model...')
    try:
        kenlm_model = kenlm.Model(check_model(model_name), **model_params)
    except:  # noqa: E722
        kenlm_model = kenlm.Model(check_model(model_name, force=True),
                                  **model_params)
    return kenlm_model


def prepare_nltk_model(lang, name_pattern='punkt.{}.pickle', **model_params):
    """
    Prepare and load a nltk punkt model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    model_params.pop('device', None)

    nltk_to_punkt = {
        'en': 'english',
        'fr': 'french',
        'pt': 'portuguese',
        'es': 'spanish'
    }
    assert lang in nltk_to_punkt.keys(
    ), 'lang must be one of the following: {}'.format(
        list(nltk_to_punkt.keys()))
    model_name = name_pattern.format(nltk_to_punkt[lang])

    logger.info('Loading nltk punkt split model...')
    try:
        nltk_model = nltk.data.load(check_model(model_name), **model_params)
    except:  # noqa: E722
        nltk_model = nltk.data.load(check_model(model_name, force=True),
                                    **model_params)
    return nltk_model


def prepare_opencv_classifier(model_path, **model_params):
    model = cv2.CascadeClassifier(model_path)
    return model


def prepare_recognizeAnything_model(
        pretrained_model_name_or_path='ram_plus_swin_large_14m.pth',
        input_size=384,
        **model_params):
    """
    Prepare and load recognizeAnything model.

    :param model_name: input model name.
    :param input_size: the input size of the model.
    """
    logger.info('Loading recognizeAnything model...')

    try:
        model = ram.ram_plus(
            pretrained=check_model(pretrained_model_name_or_path),
            image_size=input_size,
            vit='swin_l')
    except (RuntimeError, UnpicklingError) as e:  # noqa: E722
        logger.warning(e)
        model = ram.ram_plus(pretrained=check_model(
            pretrained_model_name_or_path, force=True),
                             image_size=input_size,
                             vit='swin_l')
    device = model_params.pop('device', 'cpu')
    model.to(device).eval()
    return model


def prepare_sentencepiece_model(model_path, **model_params):
    """
    Prepare and load a sentencepiece model.

    :param model_path: input model path
    :return: model instance
    """
    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_path))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_path, force=True))
    return sentencepiece_model


def prepare_sentencepiece_for_lang(lang,
                                   name_pattern='{}.sp.model',
                                   **model_params):
    """
    Prepare and load a sentencepiece model for specific langauge.

    :param lang: language to render model name
    :param name_pattern: pattern to render the model name
    :return: model instance.
    """

    model_name = name_pattern.format(lang)
    return prepare_sentencepiece_model(model_name)


def prepare_simple_aesthetics_model(pretrained_model_name_or_path,
                                    *,
                                    return_model=True,
                                    **model_params):
    """
    Prepare and load a simple aesthetics model.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    if 'device' in model_params:
        model_params['device_map'] = model_params.pop('device')

    processor = transformers.CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path, **model_params)
    if not return_model:
        return processor
    else:
        if 'v1' in pretrained_model_name_or_path:
            model = aes_pre.AestheticsPredictorV1.from_pretrained(
                pretrained_model_name_or_path, **model_params)
        elif ('v2' in pretrained_model_name_or_path
              and 'linear' in pretrained_model_name_or_path):
            model = aes_pre.AestheticsPredictorV2Linear.from_pretrained(
                pretrained_model_name_or_path, **model_params)
        elif ('v2' in pretrained_model_name_or_path
              and 'relu' in pretrained_model_name_or_path):
            model = aes_pre.AestheticsPredictorV2ReLU.from_pretrained(
                pretrained_model_name_or_path, **model_params)
        else:
            raise ValueError(
                'Not support {}'.format(pretrained_model_name_or_path))
        return (model, processor)


def prepare_spacy_model(lang,
                        name_pattern='{}_core_web_md-3.7.0',
                        **model_params):
    """
    Prepare spacy model for specific language.

    :param lang: language of sapcy model. Should be one of ["zh",
        "en"]
    :return: corresponding spacy model
    """
    import spacy

    assert lang in ['zh', 'en'], 'Diversity only support zh and en'
    model_name = name_pattern.format(lang)
    logger.info(f'Loading spacy model [{model_name}]...')
    compressed_model = '{}.tar.gz'.format(model_name)

    # decompress the compressed model if it's not decompressed
    def decompress_model(compressed_model_path):
        if not compressed_model_path.endswith('.tar.gz'):
            raise ValueError('Only .tar.gz files are supported')

        decompressed_model_path = compressed_model_path.replace('.tar.gz', '')
        if os.path.exists(decompressed_model_path) \
                and os.path.isdir(decompressed_model_path):
            return decompressed_model_path

        ver_name = os.path.basename(decompressed_model_path)
        unver_name = ver_name.rsplit('-', maxsplit=1)[0]
        target_dir_in_archive = f'{ver_name}/{unver_name}/{ver_name}/'

        import tarfile
        with tarfile.open(compressed_model_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.startswith(target_dir_in_archive):
                    # relative path without unnecessary directory levels
                    relative_path = os.path.relpath(
                        member.name, start=target_dir_in_archive)
                    target_path = os.path.join(decompressed_model_path,
                                               relative_path)

                    if member.isfile():
                        # ensure the directory exists
                        target_directory = os.path.dirname(target_path)
                        os.makedirs(target_directory, exist_ok=True)
                        # for files, extract to the specific location
                        with tar.extractfile(member) as source:
                            with open(target_path, 'wb') as target:
                                target.write(source.read())
        return decompressed_model_path

    try:
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model)))
    except:  # noqa: E722
        diversity_model = spacy.load(
            decompress_model(check_model(compressed_model, force=True)))
    return diversity_model


def prepare_video_blip_model(pretrained_model_name_or_path,
                             *,
                             return_model=True,
                             **model_params):
    """
    Prepare and load a video-clip model with the correspoding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param trust_remote_code: passed to transformers
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    if 'device' in model_params:
        model_params['device_map'] = model_params.pop('device')

    class VideoBlipVisionModel(transformers.Blip2VisionModel):
        """A simple, augmented version of Blip2VisionModel to handle
        videos."""

        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[tuple,
                   transformers.modeling_outputs.BaseModelOutputWithPooling]:
            """Flatten `pixel_values` along the batch and time dimension,
            pass it through the original vision model,
            then unflatten it back.

            :param pixel_values: a tensor of shape
            (batch, channel, time, height, width)

            :returns:
                last_hidden_state: a tensor of shape
                (batch, time * seq_len, hidden_size)
                pooler_output: a tensor of shape
                (batch, time, hidden_size)
                hidden_states:
                    a tuple of tensors of shape
                    (batch, time * seq_len, hidden_size),
                    one for the output of the embeddings +
                    one for each layer
                attentions:
                    a tuple of tensors of shape
                    (batch, time, num_heads, seq_len, seq_len),
                    one for each layer
            """
            if pixel_values is None:
                raise ValueError('You have to specify pixel_values')

            batch, _, time, _, _ = pixel_values.size()

            # flatten along the batch and time dimension to create a
            # tensor of shape
            # (batch * time, channel, height, width)
            flat_pixel_values = pixel_values.permute(0, 2, 1, 3,
                                                     4).flatten(end_dim=1)

            vision_outputs: transformers.modeling_outputs.BaseModelOutputWithPooling = super(  # noqa: E501
            ).forward(
                pixel_values=flat_pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # now restore the original dimensions
            # vision_outputs.last_hidden_state is of shape
            # (batch * time, seq_len, hidden_size)
            seq_len = vision_outputs.last_hidden_state.size(1)
            last_hidden_state = vision_outputs.last_hidden_state.view(
                batch, time * seq_len, -1)
            # vision_outputs.pooler_output is of shape
            # (batch * time, hidden_size)
            pooler_output = vision_outputs.pooler_output.view(batch, time, -1)
            # hidden_states is a tuple of tensors of shape
            # (batch * time, seq_len, hidden_size)
            hidden_states = (tuple(
                hidden.view(batch, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states)
                             if vision_outputs.hidden_states is not None else
                             None)
            # attentions is a tuple of tensors of shape
            # (batch * time, num_heads, seq_len, seq_len)
            attentions = (tuple(
                hidden.view(batch, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions)
                          if vision_outputs.attentions is not None else None)
            if return_dict:
                return transformers.modeling_outputs.BaseModelOutputWithPooling(  # noqa: E501
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=hidden_states,
                    attentions=attentions,
                )
            return (last_hidden_state, pooler_output, hidden_states,
                    attentions)

    class VideoBlipForConditionalGeneration(
            transformers.Blip2ForConditionalGeneration):

        def __init__(self, config: transformers.Blip2Config) -> None:
            # HACK: we call the grandparent super().__init__() to bypass
            # transformers.Blip2ForConditionalGeneration.__init__() so we can
            # replace self.vision_model
            super(transformers.Blip2ForConditionalGeneration,
                  self).__init__(config)

            self.vision_model = VideoBlipVisionModel(config.vision_config)

            self.query_tokens = nn.Parameter(
                torch.zeros(1, config.num_query_tokens,
                            config.qformer_config.hidden_size))
            self.qformer = transformers.Blip2QFormerModel(
                config.qformer_config)

            self.language_projection = nn.Linear(
                config.qformer_config.hidden_size,
                config.text_config.hidden_size)
            if config.use_decoder_only_language_model:
                language_model = transformers.AutoModelForCausalLM.from_config(
                    config.text_config)
            else:
                language_model = transformers.AutoModelForSeq2SeqLM.from_config(  # noqa: E501
                    config.text_config)
            self.language_model = language_model

            # Initialize weights and apply final processing
            self.post_init()

    processor = transformers.AutoProcessor.from_pretrained(
        pretrained_model_name_or_path, **model_params)
    if return_model:
        model_class = VideoBlipForConditionalGeneration
        model = model_class.from_pretrained(pretrained_model_name_or_path,
                                            **model_params)
    return (model, processor) if return_model else processor


def prepare_vllm_model(pretrained_model_name_or_path, **model_params):
    """
    Prepare and load a HuggingFace model with the correspoding processor.

    :param pretrained_model_name_or_path: model name or path
    :param model_params: LLM initialization parameters.
    :return: a tuple of (model, tokenizer)
    """
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    if model_params.get('device', '').startswith('cuda:'):
        model_params['device'] = 'cuda'

    model = vllm.LLM(model=pretrained_model_name_or_path, **model_params)
    tokenizer = model.get_tokenizer()

    return (model, tokenizer)


MODEL_FUNCTION_MAPPING = {
    'api': prepare_api_model,
    'diffusion': prepare_diffusion_model,
    'fasttext': prepare_fasttext_model,
    'huggingface': prepare_huggingface_model,
    'kenlm': prepare_kenlm_model,
    'nltk': prepare_nltk_model,
    'opencv_classifier': prepare_opencv_classifier,
    'recognizeAnything': prepare_recognizeAnything_model,
    'sentencepiece': prepare_sentencepiece_for_lang,
    'simple_aesthetics': prepare_simple_aesthetics_model,
    'spacy': prepare_spacy_model,
    'video_blip': prepare_video_blip_model,
    'vllm': prepare_vllm_model,
}

_MODELS_WITHOUT_FILE_LOCK = {
    'kenlm', 'nltk', 'recognizeAnything', 'sentencepiece', 'spacy'
}


def prepare_model(model_type, **model_kwargs):
    assert (model_type in MODEL_FUNCTION_MAPPING.keys()
            ), 'model_type must be one of the following: {}'.format(
                list(MODEL_FUNCTION_MAPPING.keys()))
    model_func = MODEL_FUNCTION_MAPPING[model_type]
    model_key = partial(model_func, **model_kwargs)
    if model_type in _MODELS_WITHOUT_FILE_LOCK:
        # initialize once in the main process to safely download model files
        model_key()
    return model_key


def get_model(model_key=None, rank=None, use_cuda=False):
    if model_key is None:
        return None

    global MODEL_ZOO
    if model_key not in MODEL_ZOO:
        logger.debug(
            f'{model_key} not found in MODEL_ZOO ({mp.current_process().name})'
        )
        if use_cuda:
            rank = rank if rank is not None else 0
            rank = rank % cuda_device_count()
            device = f'cuda:{rank}'
        else:
            device = 'cpu'
        MODEL_ZOO[model_key] = model_key(device=device)
    return MODEL_ZOO[model_key]


def free_models():
    global MODEL_ZOO
    for model_key in MODEL_ZOO:
        try:
            MODEL_ZOO[model_key].to('cpu')
        except Exception:
            pass
    MODEL_ZOO.clear()
