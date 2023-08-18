import os

import pandas as pd

from data_juicer.utils.constant import Fields
class OverallAnalysis:
    """Apply analysis on the overall stats, including mean, std, quantiles,
    etc."""

    def __init__(self, dataset, output_path):
        """
        Initialization method.

        :param dataset: the dataset to be analysed
        :param output_path: path to store the analysis results.
        """
        self.stats = pd.DataFrame(dataset[Fields.stats])
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # default percentiles to analyse
        self.default_percentiles = [0.25, 0.5, 0.75]

    def analyse(self, percentiles=[]):
        """
        Apply overall analysis on the whole dataset based on the describe
        method of pandas.

        :param percentiles: percentiles to analyse
        :return: the overall analysis result.
        """
        # merge default and customized percentiles and get overall information
        percentiles = list(set(percentiles + self.default_percentiles))
        overall = self.stats.describe(percentiles=percentiles, include='all')

        # export to result report file
        overall.to_csv(os.path.join(self.output_path, 'overall.csv'))
        overall.to_markdown(os.path.join(self.output_path, 'overall.md'))

        return overall
