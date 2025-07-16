__version__ = "1.4.1"

import os
import subprocess
import sys

from loguru import logger

# allow loading truncated images for some too large images.
from PIL import ImageFile

from data_juicer.utils.availability_utils import _is_package_available
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# For now, only INFO will be shown. Later the severity level will be changed
# when setup_logger is called to initialize the logger.
logger.remove()
logger.add(sys.stderr, level="INFO")


def _cuda_device_count():
    _torch_available = _is_package_available("torch")

    if _torch_available:
        return torch.cuda.device_count()

    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        all_devices = nvidia_smi_output.strip().split("\n")

        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES is ignored when torch is unavailable. " "All detected GPUs will be used."
            )

        return len(all_devices)
    except Exception:
        # nvidia-smi not found or other error
        return 0


def cuda_device_count():
    return _cuda_device_count()


def is_cuda_available():
    return cuda_device_count() > 0
