import unittest
import time
from loguru import logger
from data_juicer.core import Monitor
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@unittest.skip('random resource utilization fluctuation may cause failure')
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
                                        args=(10 ** 3, 2, 4,),
                                        sample_interval=0.3)
        analysis1 = Monitor.analyze_single_resource_util(dict1)
        logger.info(analysis1['resource_analysis'])

        _, dict2 = Monitor.monitor_func(self._increase_mem_func,
                                        args=(10 ** 7, 2, 4,),
                                        sample_interval=0.3)
        analysis2 = Monitor.analyze_single_resource_util(dict2)
        logger.info(analysis2['resource_analysis'])

        self.assertLessEqual(
            analysis1['resource_analysis']['Mem. util.']['avg'],
            analysis2['resource_analysis']['Mem. util.']['avg'])


if __name__ == '__main__':
    unittest.main()
