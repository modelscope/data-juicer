import unittest
import time
from loguru import logger
from data_juicer.core import Monitor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class MonitorTest(DataJuicerTestCaseBase):

    def test_monitor_current_resources(self):
        resource_dict = Monitor.monitor_current_resources()
        logger.info(resource_dict)
        self.assertIn('timestamp', resource_dict)
        self.assertIn('CPU count', resource_dict)
        self.assertIn('Mem. util.', resource_dict)

    def test_analyze_resource_util_list(self):
        resource_samples = []
        for i in range(5):
            resource_samples.append(Monitor.monitor_current_resources())
            time.sleep(0.2)
        resource_util_list = [{
            'time': 1,
            'resource': resource_samples,
        }]
        analysis_res = Monitor.analyze_resource_util_list(resource_util_list)
        logger.info(analysis_res)
        item = analysis_res[0]
        self.assertIn('resource_analysis', item)
        resource_analysis = item['resource_analysis']
        cpu_util = resource_analysis['CPU util.']
        self.assertIn('max', cpu_util)
        self.assertIn('min', cpu_util)
        self.assertIn('avg', cpu_util)

    def _increase_mem_func(self, init_len=100000, multiplier=2, times=10, interval=0.5):
        lst = list(range(init_len))
        for i in range(times):
            lst = lst * multiplier
            time.sleep(interval)

    def test_monitor_func(self):
        _, dict1 = Monitor.monitor_func(self._increase_mem_func,
                                        args=(10 ** 5, 2, 8,),
                                        sample_interval=0.3)
        resource1 = dict1['resource']
        self.assertLessEqual(resource1[1]['Mem. util.'], resource1[-2]['Mem. util.'])

        _, dict2 = Monitor.monitor_func(self._increase_mem_func,
                                        args=(10 ** 6, 2, 5,),
                                        sample_interval=0.2)
        resource2 = dict2['resource'][:]
        self.assertLessEqual(resource2[1]['Mem. util.'], resource2[-2]['Mem. util.'])

        _, dict3 = Monitor.monitor_func(self._increase_mem_func,
                                        args=(25600000, 2, 4,),
                                        sample_interval=0.3)
        resource3 = dict3['resource'][:]
        self.assertLessEqual(resource3[1]['Mem. util.'], resource3[-2]['Mem. util.'])
        self.assertGreaterEqual(resource3[1]['Mem. util.'],
                                resource1[-2]['Mem. util.'])


if __name__ == '__main__':
    unittest.main()
