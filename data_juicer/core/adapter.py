from datasets import concatenate_datasets
from loguru import logger

from data_juicer.core.monitor import Monitor


class Adapter:

    def __init__(self, cfg):
        self.cfg = cfg
        self.current_resources = Monitor.monitor_current_resources()

    @staticmethod
    def execute_and_probe(dataset, operators, sample_interval=0.5):
        """
        Process the input dataset and probe related information for each OP in
        the specified operator list.

        For now, we support the following targets to probe:
        "resource": resource utilization for each OP.
        "speed": average processing speed for each OP.

        The probe result is a list and each item in the list is the probe
        result for each OP.
        """
        if operators is None or len(operators) == 0:
            return []

        # number of test samples
        sample_num = len(dataset)
        # resource utilization list
        resource_util_list = []
        # probe for each OP
        for op in operators:

            # run single op and monitor the resource utilization
            dataset, resource_util_per_op = Monitor.monitor_func(
                op.run, args=(dataset, ), sample_interval=sample_interval)

            # calculate speed
            resource_util_per_op[
                'speed'] = sample_num / resource_util_per_op['time']
            resource_util_list.append(resource_util_per_op)

        return dataset, resource_util_list

    def workloads_adapt(self, dataset, operators):
        """
        Manage the scheduling and load balancing for the dataset processing.

        :param dataset: The dataset that needs to be processed
        :param operators: Operators in the data recipe
        """
        res_data_batches = []
        load_factor = 1.0
        left_dataset = dataset
        while left_dataset:
            data_batch, left_dataset = self.take_batch(left_dataset, self.cfg,
                                                       load_factor)
            res_batch, load_factor = self.probe_data_batches(
                data_batch, operators, load_factor)
            res_data_batches.append(res_batch)
        return concatenate_datasets(res_data_batches)

    def probe_data_batches(self, data_batch, operators, load_factor=None):
        """
        Perform small batch pre-execution to probe available resources,
        current load and estimated OP speed, returning load factors and speed
        ranks for each OP.

        :param data_batch: The dataset batch to pre-execute on
        :param operators: The OP list to be pre-execution and probe
        :param load_factor: Load factor for this batch
        :return: A list of probe results for each OP.
        """
        # process and monitor the resource utilization
        res_batch, resource_util_list = self.execute_and_probe(
            data_batch, operators)
        # analyze resource utilization
        analysis_res = Monitor.analyze_resource_util_list(resource_util_list)

        # get a new load_factor according to the analysis result
        load_factor = 1.0 * analysis_res
        pass
        logger.info(f'Adjust load factor to: {load_factor}')
        return res_batch, load_factor

    @staticmethod
    def take_batch(dataset, config, load_factor=None):
        """
        Split the dataset into batches based on configuration and load factor.

        :param dataset: The dataset to be split
        :param config: Configuration settings, including batch size
        :param load_factor: The detected load factor from pre-execution
        :return: An iterator of batches
        """
        # get initial batch size
        batch_size = config.get('batch_size', 100)
        # adapt according to the load factor
        if load_factor:
            batch_size = int(batch_size / load_factor)
            pass
        # should be in [1, 1000]
        batch_size = min(max(batch_size, 1), 1000)

        # check if there are enough samples
        num_samples = len(dataset)
        if batch_size >= num_samples:
            return dataset, None
        else:
            return dataset.take(batch_size), dataset.skip(batch_size)
