import copy
import traceback
from functools import wraps

import pyarrow as pa
from loguru import logger

from data_juicer import is_cuda_available
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import size_to_bytes
from data_juicer.utils.process_utils import calculate_np
from data_juicer.utils.registry import Registry

OPERATORS = Registry('Operators')
UNFORKABLE = Registry('Unforkable')


def convert_list_dict_to_dict_list(samples):
    # reconstruct samples from "list of dicts" to "dict of lists"
    keys = samples[0].keys()
    res_samples = {}
    for key in keys:
        res_samples[key] = [s[key] for s in samples]
    return res_samples


def convert_dict_list_to_list_dict(samples):
    # reconstruct samples from "dict of lists" to "list of dicts"
    reconstructed_samples = []
    keys = list(samples.keys())
    # take any key, since they should be of same length
    for i in range(len(samples[keys[0]])):
        reconstructed_samples.append({key: samples[key][i] for key in samples})
    return reconstructed_samples


def convert_arrow_to_python(method):

    @wraps(method)
    def wrapper(sample, *args, **kwargs):
        if isinstance(sample, pa.Table):
            sample = sample.to_pydict()
        return method(sample, *args, **kwargs)

    return wrapper


def catch_map_batches_exception(method):
    """
    For batched-map sample-level fault tolerance.
    """

    @wraps(method)
    @convert_arrow_to_python
    def wrapper(samples, *args, **kwargs):
        try:
            return method(samples, *args, **kwargs)
        except Exception as e:
            from loguru import logger
            logger.error(
                f'An error occurred in mapper operation when processing '
                f'samples {samples}, {type(e)}: {e}')
            traceback.print_exc()
            ret = {key: [] for key in samples.keys()}
            ret[Fields.stats] = []
            ret[Fields.source_file] = []
            return ret

    return wrapper


def catch_map_single_exception(method):
    """
    For single-map sample-level fault tolerance.
    The input sample is expected batch_size = 1.
    """

    def is_batched(sample):
        val_iter = iter(sample.values())
        first_val = next(val_iter)
        if not isinstance(first_val, list):
            return False
        first_len = len(first_val)
        return all(
            isinstance(val, list) and len(val) == first_len
            for val in val_iter)

    @wraps(method)
    @convert_arrow_to_python
    def wrapper(sample, *args, **kwargs):
        if is_batched(sample):
            try:
                sample = convert_dict_list_to_list_dict(sample)[0]
                res_sample = method(sample, *args, **kwargs)
                return convert_list_dict_to_dict_list([res_sample])
            except Exception as e:
                from loguru import logger
                logger.error(
                    f'An error occurred in mapper operation when processing '
                    f'sample {sample}, {type(e)}: {e}')
                traceback.print_exc()
                ret = {key: [] for key in sample.keys()}
                ret[Fields.stats] = []
                ret[Fields.source_file] = []
                return ret
        else:
            # without fault tolerance
            return method(sample, *args, **kwargs)

    return wrapper


class OP:

    _accelerator = 'cpu'
    _batched_op = False

    def __init__(self, *args, **kwargs):
        """
        Base class of operators.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        # init data keys
        self.text_key = kwargs.get('text_key', 'text')
        self.image_key = kwargs.get('image_key', 'images')
        self.audio_key = kwargs.get('audio_key', 'audios')
        self.video_key = kwargs.get('video_key', 'videos')

        # whether the model can be accelerated using cuda
        _accelerator = kwargs.get('accelerator', None)
        if _accelerator is not None:
            self.accelerator = _accelerator
        else:
            self.accelerator = self._accelerator

        # parameters to determind the number of procs for this op
        self.num_proc = kwargs.get('num_proc', None)
        self.cpu_required = kwargs.get('cpu_required', 1)
        self.mem_required = kwargs.get('mem_required', 0)
        if isinstance(self.mem_required, str):
            self.mem_required = size_to_bytes(self.mem_required) / 1024**3

        self.turbo = kwargs.get('turbo', False)

        # nested wrappers
        from data_juicer.core.data import wrap_func_with_nested_access
        for name in ['process', 'compute_stats', 'compute_hash']:
            method = getattr(self, name, None)
            if method and callable(method):
                setattr(self, f'_{name}', method)
                method = wrap_func_with_nested_access(method)
                setattr(self, name, method)

    @classmethod
    def is_batched_op(cls):
        return cls._batched_op

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.accelerator == 'cuda' and is_cuda_available()

    def runtime_np(self):
        op_proc = calculate_np(self._name, self.mem_required,
                               self.cpu_required, self.num_proc,
                               self.use_cuda())
        logger.debug(
            f'Op [{self._name}] running with number of procs:{op_proc}')
        return op_proc

    def remove_extra_parameters(self, param_dict, keys=None):
        """
            at the begining of the init of the mapper op, call
            self.remove_extra_parameters(locals())
            to get the init parameter dict of the op for convenience

        """
        if keys is None:
            param_dict = {
                k: v
                for k, v in param_dict.items() if not k.startswith('_')
            }
            param_dict.pop('self', None)
        else:
            param_dict = {k: v for k, v in param_dict.items() if k not in keys}
        return param_dict

    def add_parameters(self, init_parameter_dict, **extra_param_dict):
        """
            add parameters for each sample, need to keep extra_param_dict
            and init_parameter_dict unchanged.
        """
        related_parameters = copy.deepcopy(init_parameter_dict)
        related_parameters.update(extra_param_dict)
        return related_parameters


class Mapper(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts data editing.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Mapper, self).__init__(*args, **kwargs)

        # runtime wrappers
        if self.is_batched_op():
            self.process = catch_map_batches_exception(self.process)
        else:
            self.process = catch_map_single_exception(self.process)

    def process(self, sample):
        """
        For sample level, sample --> sample

        :param sample: sample to process
        :return: processed sample
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        new_dataset = dataset.map(
            self.process,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            desc=self._name + '_process',
        )
        if tracer:
            tracer.trace_mapper(self._name, dataset, new_dataset,
                                self.text_key)
        return new_dataset


class Filter(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that removes specific info.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Filter, self).__init__(*args, **kwargs)
        self.stats_export_path = kwargs.get('stats_export_path', None)

        # runtime wrappers
        if self.is_batched_op():
            self.compute_stats = catch_map_batches_exception(
                self.compute_stats)
        else:
            self.compute_stats = catch_map_single_exception(self.compute_stats)

    def compute_stats(self, sample, context=False):
        """
        Compute stats for the sample which is used as a metric to decide
        whether to filter this sample.

        :param sample: input sample.
        :param context: whether to store context information of intermediate
            vars in the sample temporarily.
        :return: sample with computed stats
        """
        raise NotImplementedError

    def process(self, sample):
        """
        For sample level, sample --> Boolean.

        :param sample: sample to decide whether to filter
        :return: true for keeping and false for filtering
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        if Fields.stats not in dataset.features:
            from data_juicer.core.data import add_same_content_to_new_column
            dataset = dataset.map(add_same_content_to_new_column,
                                  fn_kwargs={
                                      'new_column_name': Fields.stats,
                                      'initial_value': {}
                                  },
                                  num_proc=self.runtime_np(),
                                  desc='Adding new column for stats')
        dataset = dataset.map(self.compute_stats,
                              num_proc=self.runtime_np(),
                              with_rank=self.use_cuda(),
                              desc=self._name + '_compute_stats')
        if exporter and self.stats_export_path is not None:
            exporter.export_compute_stats(dataset, self.stats_export_path)
        new_dataset = dataset.filter(self.process,
                                     num_proc=self.runtime_np(),
                                     desc=self._name + '_process')
        if tracer:
            tracer.trace_filter(self._name, dataset, new_dataset)
        return new_dataset


class Deduplicator(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts deduplication.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Deduplicator, self).__init__(*args, **kwargs)

        # runtime wrappers
        if self.is_batched_op():
            self.compute_hash = catch_map_batches_exception(self.compute_hash)
        else:
            self.compute_hash = catch_map_single_exception(self.compute_hash)

    def compute_hash(self, sample):
        """
        Compute hash values for the sample.

        :param sample: input sample
        :return: sample with computed hash value.
        """
        raise NotImplementedError

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        dataset = dataset.map(self.compute_hash,
                              num_proc=self.runtime_np(),
                              with_rank=self.use_cuda(),
                              desc=self._name + '_compute_hash')
        show_num = tracer.show_num if tracer else 0
        new_dataset, dup_pairs = self.process(dataset, show_num)
        if tracer:
            tracer.trace_deduplicator(self._name, dup_pairs)
        return new_dataset


class Selector(OP):

    def __init__(self, *args, **kwargs):
        """
        Base class that conducts selection in dataset-level.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        """
        super(Selector, self).__init__(*args, **kwargs)

    def process(self, dataset):
        """
        Dataset --> dataset.

        :param dataset: input dataset
        :return: selected dataset.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        new_dataset = self.process(dataset)
        if tracer:
            tracer.trace_filter(self._name, dataset, new_dataset)
        return new_dataset
