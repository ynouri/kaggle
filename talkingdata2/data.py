import numpy as np
import pandas as pd
import os
# Project modules
import config
import info
import logging

column_dtypes={
    'ip': np.uint32,
    'app': np.uint16,
    'device': np.uint16,
    'os': np.uint16,
    'channel': np.uint16,
    'is_attributed': np.bool_
}

parse_dates=[
    'click_time',
    'attributed_time'
]

def load(file):
    file_path = config.DATA_PATH + file
    logging.info("File = {}".format(file_path))
    logging.info("Loading dataframe...")
    df = pd.read_csv(
        filepath_or_buffer=file_path,
        nrows=None,
        dtype=column_dtypes,
        parse_dates=parse_dates
    )
    logging.info("Dataframe loaded.")
    info.rows(df)
    info.memory(df)
    return df


def save_hdf(dataframe, original_file, suffix):
    logging.info("Saving dataframe to HDF file...")
    # If original_file = train.csv, original_name = train
    original_name = original_file.split('.')[0]
    # e.g. DATA_PATH + test_with_features.hdf
    file_path = config.DATA_PATH + original_name + '_' + suffix + '.hdf'
    logging.info("File = {}".format(file_path))
    # Write to hdf
    dataframe.to_hdf(
        path_or_buf=file_path,
        key='data',
        mode='w',
        format='f'
    )
    logging.info("Dataframe saved.")
