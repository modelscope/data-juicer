import sys
import time
from typing import Union

import fire
from loguru import logger
from pyspark.ml.feature import HashingTF, MinHashLSH, Tokenizer
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import min as mincol

from tools.distributed_deduplication.dedup_utils import (find_components,
                                                         init_spark)
from tools.quality_classifier.qc_utils import (export_result, load_dataset,
                                               tokenize_dataset)


@logger.catch
def dedup_dataset(dataset_path: str,
                  result_path: str,
                  tokenizer: Union[str, None] = None,
                  threshold: float = 0.7,
                  num_features: int = 1047576,
                  num_hashtables: int = 10,
                  text_key: str = 'text',
                  master_url: str = 'text'):
    """
    Perform fuzzy text deduplication on the given dataset.
    :param dataset_path: the path to the dataset to perform deduplication,
        The suffix of the path should be one of the json, jsonl, parquet.
    :param result_path: the path to store the predicted result dataset.
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model.
    :param threshold: if the Jaccard similarity between two documents
        exceeds a predetermined threshold, they are considered duplicates.
        The accuracy of deduplication depends on the similarity threshold set.
        The lower the threshold, the more duplicates can be identified,
        but this may also increase the risk of false positives.
        You need to adjust the threshold based on your requirements for
        deduplication accuracy.
    :param num_features: the number of features that HashingTF generates.
        Default with 1047576 as mentioned in megatron-turing-nlg paper.
    :param num_hashtables: the number of hashes used in MinHashLSH.
        Default with 10 hashes as mentioned in the GPT3 paper.
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default.
    :param master_url: the master url for spark config.
        If None, then run with local[*].
    """
    # for inited cluster,
    # provide master url such as "spark://master:7077"
    spark = init_spark(master_url=master_url)
    ds = load_dataset(spark, dataset_path, text_key=text_key)
    ds = ds.withColumn('id', F.monotonically_increasing_id()).cache()
    df = ds

    if tokenizer:
        ds = tokenize_dataset(ds, tokenizer)
    else:
        ds = Tokenizer(inputCol='text', outputCol='words').transform(ds)

    hashingTF = HashingTF(inputCol='words',
                          outputCol='features',
                          numFeatures=num_features)
    ds = hashingTF.transform(ds)

    minHash = MinHashLSH(inputCol='features',
                         outputCol='hashes',
                         numHashTables=num_hashtables)
    model = minHash.fit(ds)

    ds = model.transform(ds)

    self_join = model.approxSimilarityJoin(
        ds, ds, threshold=threshold,
        distCol='JaccardDistance').filter('datasetA.id > datasetB.id').select(
            col('datasetA.id').alias('idA'),
            col('datasetB.id').alias('idB'), col('JaccardDistance'))

    self_dup_edge = self_join.groupBy('idA').agg(
        mincol(col('idB')).alias('min_idB'))

    edges = (self_dup_edge.rdd.map(lambda row: (row.idA, row.min_idB)))

    results = find_components(edges)
    if len(results) == 0:
        logger.info('No components found.')
        sys.exit(0)

    components = spark.createDataFrame(results,
                                       schema=['id', 'component'
                                               ]).sort(['component', 'id'])
    components.show()
    df = df.join(components, on='id', how='left')
    df = df.filter(F.col('component').isNull()).drop('id', 'component').cache()
    export_result(df, result_path)


if __name__ == '__main__':
    stime = time.time()
    fire.Fire(dedup_dataset)
    etime = time.time()
    logger.info(f'Execution Done, Total time {etime - stime}')
