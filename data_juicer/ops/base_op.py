import copy
import functools
import traceback

import pandas as pd
import pyarrow as pa
from loguru import logger

from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import size_to_bytes
from data_juicer.utils.process_utils import calculate_np
from data_juicer.utils.registry import Registry

OPERATORS = Registry('Operators')


class OP:

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
        self._accelerator = kwargs.get('accelerator', 'cpu')

        # parameters to determind the number of procs for this op
        self.num_proc = kwargs.get('num_proc', 0)
        self.cpu_required = kwargs.get('cpu_required', 1)
        self.mem_required = kwargs.get('mem_required', 0)
        if isinstance(self.mem_required, str):
            self.mem_required = size_to_bytes(self.mem_required) / 1024**3

        # whether to use actor mode in ray
        self._use_actor = kwargs.get('use_actor', False)

        from data_juicer.core.data import wrap_func_with_nested_access
        self.process = wrap_func_with_nested_access(self.process)

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def runtime_np(self):
        return calculate_np(self._name, self.mem_required, self.cpu_required,
                            self.num_proc, self.use_cuda())

    def use_cuda(self):
        return self._accelerator == 'cuda'

    def use_actor(self):
        return self._use_actor

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

    def __call__(self, dataset, checkpointer=None, tracer=None):
        try:
            dataset = self.run(dataset, tracer)
            if checkpointer:
                checkpointer.record(self._name, self._process_kwargs)
            return dataset
        except:  # noqa: E722
            logger.error(f'An error occurred during Op [{self._name}].')
            traceback.print_exc()
            if checkpointer:
                logger.info('Writing checkpoint of dataset processed by '
                            'last op...')
                dataset.cleanup_cache_files()
                checkpointer.save_ckpt(dataset)
            exit(1)


def batch_mapper_wrapper(fn):

    def wrapper(samples):
        samples = samples.to_pandas()
        res = fn(samples)
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        return pa.Table.from_pandas(res)

    return wrapper


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


def catch_exception_mapper_process(method):
    """
    For mapper sample level fault tolerance.
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            samples = args[0]
            return method(*args, **kwargs)
        except Exception as e:
            samples = args[0]
            logger.error(
                f'An error occurred in mapper operation when processing '
                f'sample {samples}, {type(e)}: {e}')
            ret = {key: [] for key in samples.keys()}
            ret[Fields.stats] = []
            ret[Fields.source_file] = []
            return ret

    return wrapper


def catch_exception_mapper_process_single(method):
    """
    For mapper process_single,
    turn it into batch_size = 1, and enable fault tolerance.
    """

    def wrapper(*args, **kwargs):
        try:
            args = list(args)
            samples = args[0]
            sample = convert_dict_list_to_list_dict(samples)[0]
            args[0] = sample
            args = tuple(args)
            res_sample = method(*args, **kwargs)
            return convert_list_dict_to_dict_list([res_sample])
        except Exception as e:
            samples = args[0]
            logger.error(
                f'An error occurred in mapper operation when processing '
                f'sample {samples}, {type(e)}: {e}')
            ret = {key: [] for key in samples.keys()}
            ret[Fields.stats] = []
            ret[Fields.source_file] = []
            return ret

    return wrapper


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

        # In default, it's a normal OP instead of batched OP
        self._batched_op = kwargs.get('batched_op', False)

    def process(self, sample):
        """
        For sample level, sample --> sample

        :param sample: sample to process
        :return: processed sample
        """
        raise NotImplementedError

    def is_batched_op(self):
        return self._batched_op

    def run(self, dataset, tracer=None):
        if self._batched_op:
            batched_process = batch_mapper_wrapper(self.process)
            wrapped_process = catch_exception_mapper_process(batched_process)
        else:
            wrapped_process = catch_exception_mapper_process_single(
                self.process)
        new_dataset = dataset.map(
            wrapped_process,
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

        from data_juicer.core.data import wrap_func_with_nested_access
        self.compute_stats = wrap_func_with_nested_access(self.compute_stats)
        self.stats_export_path = kwargs.get('stats_export_path', None)

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

    def run(self, dataset, tracer=None):
        if Fields.stats not in dataset.features:
            from data_juicer.core.data import add_same_content_to_new_column
            dataset = dataset.map(add_same_content_to_new_column,
                                  fn_kwargs={
                                      'new_column_name': Fields.stats,
                                      'initial_value': {}
                                  },
                                  num_proc=self.runtime_np(),
                                  desc='Adding new column for stats')
        wrapped_process = catch_exception_mapper_process_single(
            self.compute_stats)
        dataset = dataset.map(wrapped_process,
                              num_proc=self.runtime_np(),
                              with_rank=self.use_cuda(),
                              desc=self._name + '_compute_stats')
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

        from data_juicer.core.data import wrap_func_with_nested_access
        self.compute_hash = wrap_func_with_nested_access(self.compute_hash)

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

    def run(self, dataset, tracer=None):
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

    def run(self, dataset, tracer=None):
        new_dataset = self.process(dataset)
        if tracer:
            tracer.trace_filter(self._name, dataset, new_dataset)
        return new_dataset
