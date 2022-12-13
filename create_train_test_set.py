import os
import sys
import time
from pathlib import Path

import click
import psutil
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, monotonically_increasing_id, lit, row_number, rand
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType

import random
import numpy as np
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import array, create_map, struct


RANDOM_SEED = 9876


def top_n_per_group(spark_df, groupby, topn):
    spark_df = spark_df.withColumn('rand', rand(seed=RANDOM_SEED))
    window = Window.partitionBy(col(groupby)).orderBy(col('rand'))

    return (
        spark_df
            .select(col('*'), row_number().over(window).alias('row_number'))
            .where(col('row_number') <= topn)
            .drop('row_number', 'rand')
    )


def split_train_test(df, test_size, class_balancing, c, N, k):
    # add increasing id for df
    df = df.withColumn('id', monotonically_increasing_id())

    # stratified split
    fractions = (
        df
            .select('label')
            .distinct().
            withColumn('fraction', lit(test_size))
            .rdd
            .collectAsMap()
    )
    test_id = (
        df
            .sampleBy('label', fractions, seed=RANDOM_SEED)
            .select('id')
            .withColumn('is_test', lit(True))
    )

    df = df.join(test_id, how='left', on='id')

    train_df = (
        df
            .filter(col('is_test').isNull())
            .select('feature', 'label')
    )
    test_df = (
        df.
            filter(col('is_test'))
            .select('feature', 'label')
    )

    # under sampling
    if class_balancing == 'under_sampling':
        print('Balance dataset using', class_balancing)
        # get label list with count of each label
        label_count_df = (
            train_df.
                groupby('label')
                .count()
                .toPandas()
        )

        # get min label count in train set for under sampling
        min_label_count = int(label_count_df['count'].min())

        train_df = top_n_per_group(train_df, 'label', min_label_count)

    elif class_balancing == 'SMOTE':
        print('Balance dataset using', class_balancing)

        # Change the column feature to VectorUDT for spark ml
        train_df = train_df.withColumn("feature", array_to_vector("feature"))

        # Apply SMOTE for minority labels
        train_df = smote(train_df, minority_label=0, N=N, k=k, seed=RANDOM_SEED)

        # Change the feature column back to array
        train_df = train_df.withColumn("feature", vector_to_array("feature"))

    elif class_balancing == 'SMOTE+under_sampling':
        print('Balance dataset using', class_balancing)

        # get label list with count of each label
        label_count_df = train_df.groupby('label').count().toPandas()

        # get a "c" number of minority classes to apply SMOTE
        min_classes = label_count_df.sort_values(by='count').head(c)['label'].values

        # Change the column feature to VectorUDT for spark ml
        train_df = train_df.withColumn("feature", array_to_vector("feature"))

        # Apply SMOTE for minority labels
        for label in min_classes:
            print("Applying SMOTE: min_label", label , ', N=', N, ', k=', k)
            train_df = smote(train_df, minority_label=label, N=N, k=k, seed=RANDOM_SEED)

        # Change the feature column back to array
        train_df = train_df.withColumn("feature", vector_to_array("feature"))

        # get min label count in train set for under sampling
        min_label_count = int(label_count_df['count'].min())

        # Apply undersampling to the minority class
        train_df = top_n_per_group(train_df, 'label', min_label_count)

    else:
        print('Not using any balancing technique')

    return train_df, test_df


def save_parquet(df, path):
    output_path = path.absolute().as_uri()
    (
        df
            .write
            .mode('overwrite')
            .parquet(output_path)
    )


def create_train_test_for_task(df, label_col, test_size, class_balancing, c, N, k, skip_test, train_data_dir, test_data_dir):

    task_df = df.filter(col(label_col).isNotNull()).selectExpr('feature', f'{label_col} as label')

    print('splitting train test')
    train_df, test_df = split_train_test(task_df, test_size, class_balancing, c, N, k)
    print('splitting train test done')

    train_path = train_data_dir / 'train.parquet'

    print('saving train split to:', train_path)
    save_parquet(train_df, train_path)
    print('saving train split done')

    if not skip_test:

        test_path = test_data_dir / 'test.parquet'
        print('saving test split to:', test_path)

        save_parquet(test_df, test_path)
        print('saving test split done')

    else:
        print('skip saving test split')


def print_df_label_distribution(spark, path):
    print(path)
    print(spark.read.parquet(path.absolute().as_uri()).groupby('label').count().toPandas())

################################################################
# SMOTE implementation
# Code adapted from from https://medium.com/@haoyunlai/smote-implementation-in-pyspark-76ec4ffa2f1d
################################################################
def smote(vectorized_sdf, minority_label, N, k, seed, bucketLength=1.0):
    '''
    contains logic to perform smote oversampling, given a spark df with 2 classes
    inputs:
    * vectorized_sdf: cat cols are already stringindexed, num cols are assembled into 'features' vector
      df target col should be 'label'
    * smote_config: config obj containing smote parameters
    output:
    * oversampled_df: spark df after smote oversampling
    '''
    # Find the minority class
    dataInput_min = vectorized_sdf[vectorized_sdf['label'] == minority_label]
    
    # LSH, bucketed random projection
    brp = BucketedRandomProjectionLSH(inputCol="feature", outputCol="hashes", seed=seed, bucketLength=bucketLength)

    # smote only applies on existing minority instances
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)

    # here distance is calculated from brp's param inputCol
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")

    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)

    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")

    self_similarity_df = self_join_w_distance.withColumn("r_num", F.row_number().over(over_original_rows))

    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= k)

    over_original_rows_no_order = Window.partitionBy('datasetA')

    # list to store batches of synthetic data
    res = []
    
    # two udf for vector add and subtract, subtraction include a random factor [0,1]
    subtract_vector_udf = F.udf(lambda arr: random.uniform(0, 1)*(arr[0]-arr[1]), VectorUDT())
    add_vector_udf = F.udf(lambda arr: arr[0]+arr[1], VectorUDT())
    
    # retain original columns
    original_cols = dataInput_min.columns
    
    for i in range(N):
        print("generating batch %s of synthetic instances"%i, "for minority_label", minority_label)
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected.withColumn("rand", F.rand()).withColumn('max_rand', F.max('rand').over(over_original_rows_no_order))\
                            .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel.select('*', subtract_vector_udf(F.array('datasetA.feature', 'datasetB.feature')).alias('vec_diff'))
        df_vec_modified = df_vec_diff.select('*', add_vector_udf(F.array('datasetA.feature', 'vec_diff')).alias('feature'))
        
        # for categorical cols, either pick original or the neighbour's cat values
        for c in original_cols:
            # randomly select neighbour or original data
            col_sub = random.choice(['datasetA','datasetB'])
            val = "{0}.{1}".format(col_sub,c)
            if c != 'feature':
                # do not unpack original numerical features
                df_vec_modified = df_vec_modified.withColumn(c,F.col(val))
        
        # this df_vec_modified is the synthetic minority instances,
        df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        
        res.append(df_vec_modified)
    
    dfunion = reduce(DataFrame.unionAll, res)
    # union synthetic instances with original full (both minority and majority) df
    oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    
    return oversampled_df


@click.command()
@click.option('--source', help='directory containing the preprocessed files', required=True)
@click.option('--train', help='directory for persisting train split', required=True)
@click.option('--test', help='directory for persisting test split', required=True)
@click.option('--test_size', default=0.2, help='size of test size', type=float)
@click.option('--class_balancing', help='class balancing technique for the training data')
@click.option('--skip_test', default=False, help='whether to skip generating the test split')
@click.option('--c', default=2, help='number of minority classes to apply SMOTE')
@click.option('--n', default=1, help='multiplying factor of SMOTE to generate synthetic samples')
@click.option('--k', default=5, help='number of nearest neighbors to be used in SMOTE')

def main(source, train, test, test_size, class_balancing, skip_test, c, n, k):
    prog_start = time.time()
    print("[create_test_train_set] Starting at ", time.strftime("%c %z", time.localtime(prog_start)))

    # prepare the directories for the train/test splits
    source_data_dir = Path(source)
    train_data_dir = Path(train)
    test_data_dir = Path(test)
    train_app_data_dir = train_data_dir / 'application_classification'
    test_app_data_dir = test_data_dir / 'application_classification'
    train_traffic_data_dir = train_data_dir / 'traffic_classification'
    test_traffic_data_dir = test_data_dir / 'traffic_classification'

    # initialise local spark
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    memory_gb = psutil.virtual_memory().available // 1024 // 1024 // 1024
    spark = (
        SparkSession
            .builder
            .master('local[*]')
            .config('spark.driver.memory', f'{memory_gb}g')
            .config('spark.driver.host', '127.0.0.1')
            .getOrCreate()
    )

    # define a scheme for the input data
    schema = StructType([
        StructField('app_label', LongType(), True),
        StructField('traffic_label', LongType(), True),
        StructField('feature', ArrayType(DoubleType()), True),
    ])

    # read the dataset as json preprocessed files
    df = spark.read.schema(schema).json(f'{source_data_dir.absolute().as_uri()}/*.json.gz')

    # prepare data for application classification and traffic classification
    print('processing application classification dataset')
    create_train_test_for_task(df=df, label_col='app_label',
                               test_size=test_size, class_balancing=class_balancing, c=c, N=n, k=k, skip_test=skip_test,
                               train_data_dir=train_app_data_dir, test_data_dir=test_app_data_dir)

    print('processing traffic classification dataset')
    create_train_test_for_task(df=df, label_col='traffic_label', 
                               test_size=test_size, class_balancing=class_balancing, c=c, N=n, k=k, skip_test=skip_test,
                               train_data_dir=train_traffic_data_dir, test_data_dir=test_traffic_data_dir)

    # print stats for application samples
    print_df_label_distribution(spark, train_app_data_dir / 'train.parquet')
    print_df_label_distribution(spark, test_app_data_dir / 'test.parquet')

    # print stats for traffic samples
    print_df_label_distribution(spark, train_traffic_data_dir / 'train.parquet')
    print_df_label_distribution(spark, test_traffic_data_dir / 'test.parquet')

    print("[create_test_train_set] Done in", time.strftime("%H:%M:%S", time.gmtime(time.time() - prog_start)))


################################################################
# Main program
################################################################
if __name__ == '__main__':
    main()

