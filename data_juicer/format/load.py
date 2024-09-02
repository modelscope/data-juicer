from .formatter import BaseFormatter
from .mixture_formatter import MixtureFormatter


def load_formatter(dataset_path,
                   generated_dataset_config=None,
                   text_keys=None,
                   suffixes=[],
                   add_suffix=False,
                   **kwargs) -> BaseFormatter:
    """
    Load mixture formatter for multiple different data formats with an optional
    weight(default 1.0) according to their formats.

    :param dataset_path: path to a dataset file or a dataset directory
    :param generated_dataset_config: Configuration used to create a dataset.
        The dataset will be created from this configuration if provided.
        It must contain the `type` field to specify the dataset name.
    :param text_keys: key names of field that stores sample text.
        Default: None
    :param suffixes: files with specified suffixes to be processed.
    :param add_suffix: whether to add the file suffix to dataset meta
        info
    :return: a dataset formatter.
    """
    if generated_dataset_config:
        assert isinstance(generated_dataset_config,
                          dict) and 'type' in generated_dataset_config
        args = generated_dataset_config.copy()
        obj_name = args.pop('type')
        args.update(kwargs)

        from .formatter import FORMATTERS
        return FORMATTERS.modules[obj_name](**args)

    formatter = MixtureFormatter(dataset_path=dataset_path,
                                 text_keys=text_keys,
                                 suffixes=suffixes,
                                 add_suffix=add_suffix,
                                 **kwargs)
    return formatter
