import os

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import load_image

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

BLUR_KERNEL = {'MEAN'}

@OPERATORS.register_module('image_blur_mapper')
@LOADED_IMAGES.register_module('image_blur_mapper')
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
        :param blur_type: Type of blur kernel, including ['mean', 'box', 'gaussian'].
        :param radius: Radius of blur kernel.
        :param cover: Whether the blurred image covers the original image. If set to
             false, the blurred image will be added with the suffix '_blur' and then
             saved in the same directory.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if blur_type not in ['mean', 'box', 'gaussian']:
            raise ValueError(f'Blur_type [{blur_type}] is not supported. '
                             f'Can only be one of ["mean", "box", "gaussian"].')
        if radius < 0:
            raise ValueError(f'Radius must be >= 0.')
        
        self.p = p   
    
        from PIL import ImageFilter
        if blur_type == 'mean':
            self.blur = ImageFilter.BLUR
        elif blur_type == 'box':
            self.blur = ImageFilter.BoxBlur(radius)
        else:
            self.blur = ImageFilter.GaussianBlur(radius)

    def process(self, sample, context=False):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        images = {}
        for loaded_image_key in loaded_image_keys:
            if context and loaded_image_key in sample[Fields.context]:
                # load from context
                images[loaded_image_key] = sample[
                    Fields.context][loaded_image_key]
            else:
                if loaded_image_key not in images:
                    # avoid load the same images
                    image = load_image(loaded_image_key)
                    images[loaded_image_key] = image
                    if context:
                        # store the image data into context
                        sample[Fields.context][loaded_image_key] = image

        for index, value in enumerate(loaded_image_keys):
            if self.p < np.random.rand():
                continue
            else:
                blured_image_key = os.path.join(os.path.dirname(value), '_blured.'.join(os.path.basename(value).split('.')))
                if not os.path.exists(blured_image_key):
                    img_mode = images[value].mode
                    blured_image = images[value].convert('RGB').filter(self.blur)
                    blured_image = blured_image.convert(img_mode)
                    blured_image.save(blured_image_key)
                    if context:
                        sample[Fields.context][blured_image_key] = blured_image
                loaded_image_keys[index] = blured_image_key

        sample[self.image_key] = loaded_image_keys
        return sample