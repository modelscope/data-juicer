import os
from typing import Optional, Tuple

from data_juicer.utils.constant import Fields
from data_juicer.utils.file_utils import transfer_filename
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ...utils.lazy_loader import LazyLoader
from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_IMAGES

rembg = LazyLoader('rembg', 'rembg')
onnxruntime = LazyLoader('onnxruntime', 'onnxruntime')

OP_NAME = 'image_remove_background_mapper'


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageRemoveBackgroundMapper(Mapper):
    """
    Mapper to remove background of images
    """

    def __init__(self,
                 alpha_matting: bool = False,
                 alpha_matting_foreground_threshold: int = 240,
                 alpha_matting_background_threshold: int = 10,
                 alpha_matting_erode_size: int = 10,
                 bgcolor: Optional[Tuple[int, int, int, int]] = None,
                 *args,
                 **kwargs):
        """
        Initialization method.

        alpha_matting (bool, optional):
            Flag indicating whether to use alpha matting. Defaults to False.
        alpha_matting_foreground_threshold (int, optional):
            Foreground threshold for alpha matting. Defaults to 240.
        alpha_matting_background_threshold (int, optional):
            Background threshold for alpha matting. Defaults to 10.
        alpha_matting_erode_size (int, optional):
            Erosion size for alpha matting. Defaults to 10.
        bgcolor (Optional[Tuple[int, int, int, int]], optional):
            Background color for the cutout image. Defaults to None.
        *args (Optional[Any]): Additional positional arguments.
        **kwargs (Optional[Any]): Additional keyword arguments.

        """

        super().__init__(*args, **kwargs)
        self._init_parameters = self.remove_extra_parameters(locals())

        self.alpha_matting = alpha_matting
        self.alpha_matting_foreground_threshold = \
            alpha_matting_foreground_threshold
        self.alpha_matting_background_threshold = \
            alpha_matting_background_threshold
        self.alpha_matting_erode_size = alpha_matting_erode_size
        self.bgcolor = bgcolor

    def process_single(self, sample, context=False):
        # there is no image in this sample
        if self.image_key not in sample or \
                not sample[self.image_key]:
            return []

        if Fields.source_file not in sample or not sample[Fields.source_file]:
            sample[Fields.source_file] = sample[self.image_key]

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(sample, context,
                                                loaded_image_keys, load_image)
        processed = {}

        for image_key in loaded_image_keys:
            if image_key in processed:
                continue

            remove_image_key = transfer_filename(image_key, OP_NAME,
                                                 **self._init_parameters)
            name, _ = os.path.splitext(remove_image_key)
            remove_image_key = f'{name}.png'
            if not os.path.exists(
                    remove_image_key) or remove_image_key not in images:
                rembg_image = rembg.remove(
                    images[image_key],
                    alpha_matting=self.alpha_matting,
                    alpha_matting_foreground_threshold=self.
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=self.
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size=self.alpha_matting_erode_size,
                    bgcolor=self.bgcolor)
                rembg_image.save(remove_image_key, format='PNG')
                images[remove_image_key] = rembg_image
                if context:
                    sample[Fields.context][remove_image_key] = rembg_image
            processed[image_key] = remove_image_key

        # when the file is modified, its source file needs to be updated.
        for i, value in enumerate(loaded_image_keys):
            if sample[Fields.
                      source_file][i] != value and processed[value] != value:
                sample[Fields.source_file][i] = processed[value]
        sample[self.image_key] = [processed[key] for key in loaded_image_keys]
        return sample
