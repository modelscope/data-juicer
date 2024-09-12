import threading
import time
from functools import partial

from data_juicer.utils.resource_utils import (get_cpu_count,
                                              get_cpu_utilization,
                                              query_cuda_info, query_mem_info)


class Monitor:
    """
    Monitor resource utilization and other information during the data
    processing.

    Resource utilization dict: (for each func)
    '''python
    {
        'time': 10,
        'resource': [
            {
                'timestamp': xxx,
                'CPU count': xxx,
                'GPU free mem.': xxx.
                ...
            },
            {
                'timestamp': xxx,
                'CPU count': xxx,
                'GPU free mem.': xxx,
                ...
            },
        ]
    }
    '''

    Based on the structure above, the resource utilization analysis result will
    add several extra fields on the first level:
    '''python
    {
        'time': 10,
        'resource': [...],
        'resource_analysis': {
            'GPU free mem.': {
                'max': xxx,
                'min': xxx,
                'avg': xxx,
            },
            ...
        }
    }
    '''
    Only those fields in DYNAMIC_FIELDS will be analyzed.
    """

    DYNAMIC_FIELDS = {
        'CPU util.',
        'Used mem.',
        'Free mem.',
        'Available mem.',
        'GPU free mem.',
        'GPU used mem.',
        'GPU util.',
    }

    def __init__(self):
        pass

    def monitor_all_resources(self):
        """
        Detect the resource utilization of all distributed nodes.
        """
        # TODO
        pass

    @staticmethod
    def monitor_current_resources():
        """
        Detect the resource utilization of the current environment/machine.
        All data of "util." is in percent. All data of "mem." is in MB.
        """
        resource_dict = dict()
        # current time
        resource_dict['timestamp'] = time.time()

        # CPU
        resource_dict['CPU count'] = get_cpu_count()
        resource_dict['CPU util.'] = get_cpu_utilization()
        resource_dict['Total mem.'] = query_mem_info('total')
        resource_dict['Used mem.'] = query_mem_info('used')
        resource_dict['Free mem.'] = query_mem_info('free')
        resource_dict['Available mem.'] = query_mem_info('available')

        # GPU
        resource_dict['GPU total mem.'] = query_cuda_info('memory.total')
        resource_dict['GPU free mem.'] = query_cuda_info('memory.free')
        resource_dict['GPU used mem.'] = query_cuda_info('memory.used')
        resource_dict['GPU util.'] = query_cuda_info('utilization.gpu')

        return resource_dict

    def analyze_resource_util_list(self, resource_util_list):
        """
        Analyze the resource utilization for a given resource util list.
        Compute {'max', 'min', 'avg'} of resource metrics for each dict item.
        """
        res_list = []
        for item in resource_util_list:
            res_list.append(self.analyze_single_resource_util(item))
        return res_list

    def analyze_single_resource_util(self, resource_util_dict):
        """
        Analyze the resource utilization for a single resource util dict.
        Compute {'max', 'min', 'avg'} of each resource metrics.
        """
        analysis_res = {}
        record_list = {}
        for record in resource_util_dict['resource']:
            for key in self.DYNAMIC_FIELDS:
                if key in record:
                    record_list.setdefault(key, []).append(record[key])

        # analyze the max, min, and avg
        for key in record_list:
            analysis_res[key] = {
                'max': max(record_list[key]),
                'min': min(record_list[key]),
                'avg': sum(record_list[key]) / len(record_list),
            }
        resource_util_dict['resource_analysis'] = analysis_res

        return resource_util_dict

    @staticmethod
    def monitor_func(func, args=None, sample_interval=0.5):
        """
        Process the input dataset and probe related information for each OP in
        the specified operator list.

        For now, we support the following targets to probe:
        "resource": resource utilization for each OP.
        "speed": average processing speed for each OP.

        The probe result is a list and each item in the list is the probe
        result for each OP.
        """
        if args is None:
            args = {}
        if isinstance(args, dict):
            func = partial(func, **args)
        elif isinstance(args, list) or isinstance(args, tuple):
            func = partial(func, *args)
        else:
            func = partial(func, args)

        # resource utilization dict
        resource_util_dict = {}

        # whether in the monitoring mode
        running_flag = False

        def resource_monitor(interval):
            # function to monitor the resource
            # interval is the sampling interval
            this_states = []
            while running_flag:
                this_states.append(Monitor.monitor_current_resources())
                time.sleep(interval)
            resource_util_dict['resource'] = this_states

        # start monitor
        running_flag = True
        monitor_thread = threading.Thread(target=resource_monitor,
                                          args=(sample_interval, ))
        monitor_thread.start()
        # start timer
        start = time.time()

        # run single op
        ret = func()

        # end timer
        end = time.time()
        # stop monitor
        running_flag = False
        monitor_thread.join()

        # calculate speed
        resource_util_dict['time'] = end - start

        return ret, resource_util_dict
