import argparse
import glob
import time
import pandas as pd
import pyspark.pandas as ps
import numpy as np
from pyspark.sql.functions import col, monotonically_increasing_id, lit, row_number, rand
from pyspark.sql.types import IntegerType


if __name__ == '__main__':
    print("[SMOTE] Starting at ", time.strftime("%c %z", time.localtime(time.time())))
    prog_start = time.time()

    parser = argparse.ArgumentParser(description="SMOTE script")
    parser.add_argument("files", nargs='+', help="List of transformed pcap files in json compressed format")
    args = parser.parse_args()

    # Read the dataframe as a pandas-spark dataframe
    df = ps.read_json(args.files, lines=True)

    # Enable spark operations
    spark_df = df.to_spark()

    # Change class labels to integers
    spark_df = spark_df.withColumn("app_label", col("app_label").cast(IntegerType()))
    spark_df = spark_df.withColumn("traffic_label", col("traffic_label").cast(IntegerType()))

    # Show initial class distributions
    print(spark_df.groupBy("app_label").count().orderBy("app_label").show())
    print(spark_df.groupBy("traffic_label").count().orderBy("traffic_label").show())

    #TODO: Filter null values if necessary
    #TODO: SMOTE from scratch: https://medium.com/@corymaklin/synthetic-minority-over-sampling-technique-smote-7d419696b88c
    #TODO: SMOTE using imblearn: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

    print(f'[SMOTE] Done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - prog_start))}')
