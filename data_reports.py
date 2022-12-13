import os
import sys
import argparse
import psutil
import glob
import time
import pandas as pd
import pyspark.pandas as ps
import numpy as np
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, monotonically_increasing_id, lit, row_number, rand
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

from utils import PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID, ID_TO_APP, ID_TO_TRAFFIC


################################################################
# Utilities
################################################################
def format_numbers(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val / 1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val / 1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]

    return new_tick_format


def plot_label_dist(sdf, task, file_name):

    label_dist = sdf.groupBy("label").count().orderBy("label").toPandas()
    label_dist = label_dist.loc[~label_dist['label'].isnull()]

    if task == 'app':
        label_dist['name'] = label_dist['label'].apply(lambda x: ID_TO_APP[x])

    elif task == 'traffic':
        label_dist['name'] = label_dist['label'].apply(lambda x: ID_TO_TRAFFIC[x])

    sns.set(rc={'figure.figsize': (10, 10)})
    sns.set_context('talk', font_scale=1.6)
    fig, ax = plt.subplots()

    label_dist.sort_values(by='count').plot(ax = ax, x='name', y='count', kind='barh', stacked=True, width=0.8, legend=None)

    # Format big numbers for readability
    ax.set_ylabel('')
    ax.set_xlabel('Number of samples')
    ax.xaxis.set_major_formatter(tick.FuncFormatter(format_numbers))

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':
    prog_start = time.time()
    print("[data_reports] Starting at ", time.strftime("%c %z", time.localtime(prog_start)))
    parser = argparse.ArgumentParser(description="Data reporting program for Deep Packet")

    parser.add_argument("-p", "--parquet", help="directory containing a paquet data split")
    parser.add_argument("-t", "--task", help="app for application_classification and traffic for traffic_classification")
    parser.add_argument("-o", "--outfile", help="output file for the distribution plot")
    args = parser.parse_args()

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
        StructField('label', LongType(), True),
        StructField('feature', ArrayType(DoubleType()), True),
    ])

    # Read the dataset from a parquet directory
    sdf = spark.read.schema(schema).parquet(args.parquet)

    # Plot data distribution for the application dataset
    plot_label_dist(sdf, args.task, args.outfile)

    print("[data_reports] Done in", time.strftime("%H:%M:%S", time.gmtime(time.time() - prog_start)))
