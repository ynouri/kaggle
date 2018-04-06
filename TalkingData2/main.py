#!/usr/bin/env python3

import pandas as pd
import numpy as np
from features import add_all_features
import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

# Memory usage
def info_memory_usage(df):
    mem_gb = df.memory_usage().sum() / 1024 ** 3
    logging.info("Dataframe memory usage = {:0.2f}GB".format(mem_gb))

# Files
path = '/Users/ynouri/ipython/Kaggle-TalkingData/input/'
sample = 'train_sample.csv'
test = 'test.csv'

# Main script
if __name__ == '__main__':

    # Load dataframe
    file = path + test
    logging.info("File = {}".format(file))
    logging.info("Loading dataframe...")
    df = pd.read_csv(file)
    logging.info("Dataframe loaded.")
    logging.info("Number of rows = {}".format(len(df)))
    info_memory_usage(df)

    # Add features
    logging.info("Start adding features...")
    add_all_features(df)
    info_memory_usage(df)
