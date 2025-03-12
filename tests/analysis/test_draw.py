import unittest
import pandas as pd
import matplotlib.pyplot as plt

from data_juicer.analysis.draw import draw_heatmap

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class DrawTest(DataJuicerTestCaseBase):

    def test_basic_func(self):
        test_data = [
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 4, 'b': 5, 'c': 6},
            {'a': 7, 'b': 8, 'c': 9},
            {'a': 10, 'b': 11, 'c': 12},
            {'a': 13, 'b': 14, 'c': 15},
        ]
        data = pd.DataFrame.from_records(test_data)
        ret = draw_heatmap(data, data.columns, triangle=True, show=True)
        self.assertIsInstance(ret, plt.Figure)
        ret = draw_heatmap(data, data.columns, show=True)
        self.assertIsInstance(ret, plt.Figure)


if __name__ == '__main__':
    unittest.main()
