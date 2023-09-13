from data_juicer.utils.registry import Registry

OPERATORS = Registry('Operators')


class Mapper:

    def __init__(self, text_key: str = None):
        """
        Base class that conducts text editing.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        """
        if text_key is None:
            text_key = 'text'
        self.text_key = text_key
        from data_juicer.core.data import wrap_func_with_nested_access
        self.process = wrap_func_with_nested_access(self.process)

        # In default, it's a normal OP instead of batched OP
        self._batched_op = False

    def process(self, sample):
        """
        For sample level, sample --> sample

        :param sample: sample to process
        :return: processed sample
        """
        raise NotImplementedError

    def is_batched_op(self):
        return self._batched_op


class Filter:

    def __init__(self, text_key: str = None):
        """
        Base class that removes specific info.

        :param text_key: the key name of field that stores sample texts
            to be processed
        """
        if text_key is None:
            text_key = 'text'
        self.text_key = text_key
        from data_juicer.core.data import wrap_func_with_nested_access
        self.process = wrap_func_with_nested_access(self.process)
        self.compute_stats = wrap_func_with_nested_access(self.compute_stats)

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


class Deduplicator:

    def __init__(self, text_key: str = None):
        """
        Base class that conducts deduplication.

        :param text_key: the key name of field that stores sample texts
            to be processed
        """
        if text_key is None:
            text_key = 'text'
        self.text_key = text_key
        from data_juicer.core.data import wrap_func_with_nested_access
        self.process = wrap_func_with_nested_access(self.process)
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


class Selector:

    def __init__(self, text_key: str = None):
        """
        Base class that conducts selection in dataset-level.

        :param text_key: the key name of field that stores sample texts
            to be processed
        """
        if text_key is None:
            text_key = 'text'
        self.text_key = text_key
        from data_juicer.core.data import wrap_func_with_nested_access
        self.process = wrap_func_with_nested_access(self.process)

    def process(self, dataset):
        """
        Dataset --> dataset.

        :param dataset: input dataset
        :return: selected dataset.
        """
        raise NotImplementedError
