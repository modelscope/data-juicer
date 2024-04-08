import sys

import fire
from loguru import logger
from pyspark.ml.feature import HashingTF, MinHashLSH
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from pyspark.sql.functions import min as mincol

from tools.distributed_deduplication.dedup_utils import (find_components,
                                                         init_spark)
from tools.quality_classifier.qc_utils import (export_result, load_dataset,
                                               tokenize_dataset)


@logger.catch
def dedup_dataset(dataset_path,
                  result_path,
                  tokenizer=None,
                  num_hashtables=10,
                  num_features=1047576,
                  text_key='text'):
    spark = init_spark()
    ds = load_dataset(spark, dataset_path, text_key=text_key)
    ds = ds.withColumn('id', F.monotonically_increasing_id()).cache()
    df = ds

    if tokenizer:
        ds = tokenize_dataset(ds, tokenizer)

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
        ds, ds, threshold=0.5,
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
    # df.write.mode("overwrite").json("output_of_spark")
    export_result(df, result_path)


if __name__ == '__main__':
    fire.Fire(dedup_dataset)
