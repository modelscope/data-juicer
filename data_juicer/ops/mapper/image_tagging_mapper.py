from collections import Counter

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_tagging_mapper'

with AvailabilityChecking(
    ['torch', 'git+https://github.com/xinyu1205/recognize-anything.git'],
        OP_NAME):
    import ram  # noqa: F401
    import torch

    # avoid hanging when calling recognizeAnything in multiprocessing
    torch.set_num_threads(1)


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageTaggingMapper(Mapper):
    """Mapper to generate image tags.
    """

    _accelerator = 'cuda'

    def __init__(self, *args, **kwargs):
        """
        Initialization method.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(
            model_type='recognizeAnything',
            pretrained_model_name_or_path='ram_plus_swin_large_14m.pth',
            input_size=384)
        from ram import get_transform
        self.transform = get_transform(image_size=384)

    def process(self, sample, rank=None, context=False):
        # check if it's generated already
        if Fields.image_tags in sample:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.image_tags] = []
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        model = get_model(self.model_key, rank, self.use_cuda())
        image_tags = []
        for _, value in enumerate(loaded_image_keys):
            image = images[value]

            image_tensor = torch.unsqueeze(self.transform(image)).to(
                next(model.parameters()).device)
            with torch.no_grad():
                tags, _ = model.generate_tag(image_tensor)

            words = [word.strip() for tag in tags for word in tag.split('|')]
            word_count = Counter(words)
            sorted_word_list = [item for item, _ in word_count.most_common()]
            image_tags.append(sorted_word_list)

        sample[Fields.image_tags] = image_tags
        return sample
