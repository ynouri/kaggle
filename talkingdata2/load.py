import numpy as np
import pandas as pd
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

def data(file):
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
