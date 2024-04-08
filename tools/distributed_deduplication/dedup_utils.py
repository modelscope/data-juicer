# The Star-Graph-Connected-Components (SGCC) algorithm here referenced from:
# https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication_spark.py
# --------------------------------------------------------

from pyspark import SparkConf
from pyspark.sql import SparkSession


def init_spark():
    conf = SparkConf()
    conf.set('spark.app.name', 'MinHashLSH')
    conf.set('spark.debug.maxToStringFields', '100')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark


# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]


def find_components(edges):
    """
    Star-Graph-Connected-Components (SGCC) algorithm
    """

    a = edges
    while True:
        b = a.flatMap(large_star_map).groupByKey().flatMap(
            large_star_reduce).distinct().cache()
        a = b.map(small_star_map).groupByKey().flatMap(
            small_star_reduce).distinct().cache()
        changes = a.subtract(b).union(b.subtract(a)).collect()
        if len(changes) == 0:
            break

    results = a.collect()
    return results
