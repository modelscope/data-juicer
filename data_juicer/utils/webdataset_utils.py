import io
import json
from typing import Any, Dict, Optional, Union


def _custom_default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]] = True):
    """A custom decoder for webdataset. Support multiple images list decoding.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    sample = dict(sample)
    for key, value in sample.items():
        extension = key.split(".")[-1]
        if key.startswith("__"):
            continue
        elif extension in ["txt", "text"]:
            sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            sample[key] = int(value.decode("utf-8"))
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
            import numpy as np
            import PIL.Image

            if format == "PIL":
                sample[key] = PIL.Image.open(io.BytesIO(value))
            else:
                sample[key] = np.asarray(PIL.Image.open(io.BytesIO(value)))
        elif extension in ["jpgs", "pngs", "ppms", "pgms", "pbms", "pnms"]:
            import pickle

            import numpy as np
            import PIL.Image

            value = pickle.loads(value)

            if format == "PIL":
                sample[key] = [PIL.Image.open(io.BytesIO(v)) for v in value]
            else:
                sample[key] = [np.asarray(PIL.Image.open(io.BytesIO(v))) for v in value]
        elif extension == "json":
            sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            sample[key] = np.load(io.BytesIO(value))
        elif extension == "mp":
            import msgpack

            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch

            sample[key] = torch.load(io.BytesIO(value))
        elif extension in ["pickle", "pkl"]:
            import pickle

            sample[key] = pickle.loads(value)
    return sample


def _custom_default_encoder(sample: Dict[str, Any], format: Optional[Union[str, bool]] = True):
    """A custom encoder for webdataset.
    In addition to the original encoding, it also supports encode image lists and byte type images.

    This handles common file extensions: .txt, .cls, .cls2, .jpg,
        .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl, .jpgs (images list),
        .jpegs (images list), .pngs (images list).
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own encoder.

    Args:
        sample (Dict[str, Any]): sample
    """
    from ray.data._internal.datasource.webdataset_datasource import extension_to_format

    sample = dict(sample)
    for key, value in sample.items():
        extension = key.split(".")[-1]
        if key.startswith("__"):
            continue
        elif extension in ["txt"]:
            if isinstance(value, list):
                sample[key] = [v.encode("utf-8") for v in value]
            else:
                sample[key] = value.encode("utf-8")
        elif extension in ["cls", "cls2"]:
            sample[key] = str(value).encode("utf-8")
        elif extension in ["jpg", "jpeg", "png", "ppm", "pgm", "pbm", "pnm"]:
            import numpy as np
            import PIL.Image

            if isinstance(value, np.ndarray):
                value = PIL.Image.fromarray(value)
            elif isinstance(value, bytes):
                pass
            else:
                assert isinstance(value, PIL.Image.Image), f"{key} should be a PIL image, got {type(value)}"
            stream = io.BytesIO()
            value.save(stream, format=extension_to_format.get(extension.lower(), extension))
            sample[key] = stream.getvalue()
        elif extension in ["jpgs", "jpegs", "pngs", "ppms", "pgms", "pbms", "pnms"]:
            import numpy as np
            import PIL.Image

            def _encode_image(value):
                if isinstance(value, np.ndarray):
                    value = PIL.Image.fromarray(value)
                elif isinstance(value, bytes):
                    return value
                assert isinstance(value, PIL.Image.Image)
                stream = io.BytesIO()
                value.save(stream, format=extension_to_format.get(extension.lower(), extension))
                return stream.getvalue()

            import pickle

            sample[key] = pickle.dumps([_encode_image(v) for v in value])

        elif extension == "json":
            sample[key] = json.dumps(value).encode("utf-8")
        elif extension == "npy":
            import numpy as np

            stream = io.BytesIO()
            np.save(stream, value)
            sample[key] = stream.getvalue()
        elif extension == "mp":
            import msgpack

            sample[key] = msgpack.dumps(value)
        elif extension in ["pt", "pth"]:
            import torch

            stream = io.BytesIO()
            torch.save(value, stream)
            sample[key] = stream.getvalue()
        elif extension in ["pickle", "pkl"]:
            import pickle

            stream = io.BytesIO()
            pickle.dump(value, stream)
            sample[key] = stream.getvalue()
    return sample


def reconstruct_custom_webdataset_format(samples, field_mapping: Optional[Dict[str, str]] = None):
    """
    Reconstruct the original dataset to the WebDataset format.
    For all keys, they can be specified by `field_mapping` argument, which is a dict mapping from the target
    field key in the result format to the source field key in the original samples.

    :param samples: the input samples batch to be reconstructed
    :param field_mapping: the field mapping to construct the left fields.
    """
    if field_mapping is None:
        field_mapping = {}
    assert isinstance(field_mapping, dict)

    # not specified -- return the original samples
    if len(field_mapping) == 0:
        return samples

    # construct the left fields
    reconstructed_sample = {}
    for tgt_field, src_field in field_mapping.items():
        assert isinstance(src_field, str) or isinstance(src_field, list)
        if isinstance(src_field, str):
            reconstructed_sample[tgt_field] = samples[src_field]
        elif isinstance(src_field, list):
            reconstructed_sample[tgt_field] = {src_field_item: samples[src_field_item] for src_field_item in src_field}

    return reconstructed_sample
