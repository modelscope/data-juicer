from data_juicer.core.monitor import Monitor


class Adapter:

    def __init__(self):
        self.current_resources = self.detect_current_resources()

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
        resource_util = []
        # probe for each OP
        for op in operators:

            # run single op and monitor the resource utilization
            dataset, resource_util_per_op = Monitor.monitor_func(
                op.run, args=(dataset, ), sample_interval=sample_interval)

            # calculate speed
            resource_util_per_op[
                'speed'] = sample_num / resource_util_per_op['time']
            resource_util.append(resource_util_per_op)

        return resource_util

    def probe_small_batches(self, dataset, operators):
        """
        Perform small batch pre-execution to probe available resources,
        current load and estimated OP speed, returning load factors and speed
        ranks for each OP.

        :param dataset: The dataset to pre-execute small batches on
        :param operators: The OP list to be pre-execution and probe
        :return: A list of probe results for each OP.
        """
        # 假设这个函数执行一小部分数据以检测负载
        print('Pre-executing small batches to detect system load...')
        # 这里可以添加具体的逻辑来预执行小批量
        # 模拟的负载因子（可以根据实际情况计算）

        # 例如，在负载过高的情况下返回较小的值
        load_factor = self.available_resources['load']  # 返回当前负载
        print(f'Detected load factor: {load_factor}')
        return load_factor

    @staticmethod
    def batch_split(dataset, config, load_factor=None):
        """
        Split the dataset into batches based on configuration and load factor.

        :param dataset: The dataset to be split
        :param config: Configuration settings, including batch size
        :param load_factor: The detected load factor from pre-execution
        :return: An iterator of batches
        """
        # get initial batch size
        adjusted_batch_size = config.get('batch_size', 100)
        # adapt according to the load factor
        if load_factor:
            adjusted_batch_size = int(adjusted_batch_size / load_factor)
        # should be in [1, 1000]
        adjusted_batch_size = min(max(adjusted_batch_size, 1), 1000)

        return dataset.iter(batch_size=adjusted_batch_size)
