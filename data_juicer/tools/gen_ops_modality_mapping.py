from typing import Dict, List, Optional, Union

from data_juicer.ops import OPERATORS
from data_juicer.ops.op_fusion import (INTER_LINES, INTER_WORDS, LOADED_AUDIOS,
                                       LOADED_IMAGES, LOADED_VIDEOS)


def get_data_processing_ops(
        ops_type: Optional[str] = None,
        modality: Optional[str] = None) -> List[Dict[str, Union[str, tuple]]]:
    """
    Retrieves a list of available data processing operators based on the specified type and data modality.

    [Docstring content from previous response, updated to reflect the code]
    """
    with open('ops_modality_mapping.json', 'r') as f:
        import json

        op_modality_mapping = json.load(f)

    if modality:
        ops_list = op_modality_mapping[modality]
    else:
        ops_list = [].extend(ops for _, ops in op_modality_mapping.items())

    if ops_type:
        ops_list = [ops for ops in ops_list if ops_type in ops['name']]

    return ops_list


def get_parameter_descriptions(init_method):
    """Helper function to extract parameter descriptions from the __init__ method."""
    param_descriptions = []
    if init_method is not None:
        for param_name, param_annotation in init_method.__annotations__.items(
        ):
            if param_name == 'return':
                continue  # Skip the return annotation

            param_type_str = (str(param_annotation) if hasattr(
                param_annotation, '__name__') else str(param_annotation))
            param_descriptions.append(
                f'{param_name} ({param_type_str}): '
                f'{get_docstring_parameter(init_method.__doc__, param_name)}'
            )  # Get details from the docstring
    return param_descriptions


def get_docstring_parameter(docstring, param_name):
    """
    Helper function to extract parameter description from a function's docstring

    :param docstring: function's docstring
    :param param_name: parameter name
    :return: parameter description
    """
    import re

    if docstring is None:
        return ''

    param_pattern = (r':param\s+' + param_name +
                     r':\s+(.*?)(?=:param|:return|:raises|\Z)')
    match = re.search(param_pattern, docstring, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ''


def get_op_modality(op_name: str) -> set[str]:
    """Helper function to determine the modality of an operator."""
    modalities = set()
    if op_name in LOADED_AUDIOS.modules:
        modalities.add('audio')
    if op_name in LOADED_IMAGES.modules:
        modalities.add('image')
    if op_name in LOADED_VIDEOS.modules:
        modalities.add('video')
    if op_name in INTER_LINES.modules or op_name in INTER_WORDS.modules:
        modalities.add('text')
    # Add other modalities as needed
    if any(keyword in op_name
           for keyword in ['text', 'line', 'word']):  # Heuristic for text
        modalities.add('text')
    if any(keyword in op_name for keyword in ['image']):  # Heuristic for image
        modalities.add('image')
    if any(keyword in op_name for keyword in ['audio']):  # Heuristic for audio
        modalities.add('audio')
    if any(keyword in op_name for keyword in ['video']):  # Heuristic for video
        modalities.add('video')
    if len(modalities) == 0:
        modalities.add('unknown')
    return modalities


if __name__ == '__main__':
    op_modality = {
        'text': {},
        'image': {},
        'audio': {},
        'video': {},
        'unknown': {},
        'multimodal': {},
    }

    for (
            op_name,
            op_cls,
    ) in OPERATORS.modules.items():  # Iterate through all registered operators
        op_info = {
            op_name: (op_cls.__doc__.strip() if op_cls.__doc__ else
                      'No description provided.') + '\n' + 'Parameters:' +
            '\n'.join(tuple(get_parameter_descriptions(op_cls.__init__)))
        }
        modalities = get_op_modality(op_name)
        if len(modalities) > 1:
            op_modality['multimodal'].update(op_info)
        else:
            op_modality[list(modalities)[0]].update(op_info)

    with open('ops_modality_mapping.json', 'w') as f:
        import json

        json.dump(op_modality, f, indent=4)

    for modality, ops in op_modality.items():
        print(modality, len(ops))

    # with open("available_ops.json", "w") as f:
    #     import json
    #     text_filter = get_available_data_processing_ops("filter", "text")
    #     json.dump(text_filter, f, indent=4)
    #     print(len(text_filter))
