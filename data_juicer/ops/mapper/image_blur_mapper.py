import os

import numpy as np

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Mapper, catch_exception_mapper_process_single
from ..op_fusion import LOADED_IMAGES

OP_NAME = 'image_blur_mapper'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageBlurMapper(Mapper):
    """Mapper to blur images.
    """

    def __init__(self,
                 p: float = 0.2,
                 blur_type: str = 'gaussian',
                 radius: float = 2,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param p: Probability of the image being blured.
        :param blur_type: Type of blur kernel, including
            ['mean', 'box', 'gaussian'].
        :param radius: Radius of blur kernel.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())
        if blur_type not in ['mean', 'box', 'gaussian']:
            raise ValueError(
                f'Blur_type [{blur_type}] is not supported. '
                f'Can only be one of ["mean", "box", "gaussian"]. ')
        if radius < 0:
            raise ValueError('Radius must be >= 0. ')

        self.p = p

        from PIL import ImageFilter
        if blur_type == 'mean':
            self.blur = ImageFilter.BLUR
        elif blur_type == 'box':
            self.blur = ImageFilter.BoxBlur(radius)
        else:
            self.blur = ImageFilter.GaussianBlur(radius)

    @catch_exception_mapper_process_single
    def process(self, sample, context=False):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)

        for index, value in enumerate(loaded_image_keys):
            if self.p < np.random.rand():
                continue
            else:
                blured_image_key = transfer_filename(value, OP_NAME,
                                                     **self._init_parameters)
                if not os.path.exists(
                        blured_image_key) or blured_image_key not in images:
                    blured_image = images[value].convert('RGB').filter(
                        self.blur)
                    images[blured_image_key] = blured_image
                    blured_image.save(blured_image_key)
                    if context:
                        sample[Fields.context][blured_image_key] = blured_image
                loaded_image_keys[index] = blured_image_key

        sample[self.image_key] = loaded_image_keys
        return sample
