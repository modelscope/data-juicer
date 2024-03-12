import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def draw_heatmap(data, xlabels, ylables=None, figsize=None, triangle=False):
    """
    Draw heatmap of input data with special lables.

    :param data: input data, now support
        [`list`, `tuple`, `numpy array`, 'torch tensor']
    :param xlabels: x axis labels.
    :param ylabels: y axis labels, if None, use xlabels.
    :param figsize: figure size.
    :param triangle: only display triangle.
    :return: a plot figure.
    """
    figsize = figsize if figsize else (8 * 2.5, 6 * 2.5)
    _, ax = plt.subplots(figsize=figsize)
    mask = None
    if triangle:
        mask = np.triu(np.ones_like(data))
    ax.tick_params(
        right=True,
        top=True,
        labelright=True,
        labeltop=True,
    )
    sns.heatmap(data,
                ax=ax,
                cmap='Oranges',
                annot=True,
                mask=mask,
                linewidths=.05,
                square=True,
                xticklabels=xlabels,
                yticklabels=ylables,
                annot_kws={'size': 8})
    plt.subplots_adjust(left=.1, right=0.95, bottom=0.22, top=0.95)
    fig = plt.gcf()
    plt.show()
    return fig
