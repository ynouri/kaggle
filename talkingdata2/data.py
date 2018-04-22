import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
# Project modules
import config
import info
import logging


def get_column_dtypes(file):
    """Return column dtypes."""
    column_dtypes = {
        'ip': np.uint32,
        'app': np.uint16,
        'device': np.uint16,
        'os': np.uint16,
        'channel': np.uint16,
        'is_attributed': np.bool_
    }
    if 'test' in file:
        column_dtypes.pop('is_attributed')
    return column_dtypes


def get_parse_dates(file):
    """Return a list of columns for which dates have to be parsed."""
    parse_dates = [
        'click_time',
        'attributed_time'
    ]
    if 'test' in file:
        parse_dates.remove('attributed_time')
    return parse_dates


def load(file):
    """Load a .csv or .hdf dataset from data directory."""
    _, extension = os.path.splitext(file)
    file_path = os.path.join(config.DATA_PATH, file)
    logging.info("File = {}".format(file_path))
    logging.info("Loading dataframe...")
    if extension == '.csv':
        df = pd.read_csv(
            filepath_or_buffer=file_path,
            nrows=None,
            dtype=get_column_dtypes(file),
            parse_dates=get_parse_dates(file)
        )
    elif extension == '.hdf':
        df = pd.read_hdf(
            path_or_buf=file_path,
            key='data',
            nrows=None
        )
    logging.info("Dataframe loaded.")
    info.rows(df)
    info.memory(df)
    return df


def save_hdf(dataframe, original_file, suffix):
    """Save a dataframe to .hdf format in the data directory."""
    logging.info("Saving dataframe to HDF file...")
    # If original_file = train.csv, original_name = train
    original_name = original_file.split('.')[0]
    # e.g. DATA_PATH + test_with_features.hdf
    file = original_name + '_' + suffix + '.hdf'
    file_path = os.path.join(config.DATA_PATH, file)
    logging.info("File = {}".format(file_path))
    # Write to hdf
    dataframe.to_hdf(
        path_or_buf=file_path,
        key='data',
        mode='w',
        format='f'
    )
    logging.info("Dataframe saved.")


def persist_dump(object_to_dump):
    """
    Dump sklearn objects (scaler, logreg) to disk.

    Name of dumped file is equal to the class name with .pkl extension.
    E.g. StandardScaler.pkl, LogisticRegression.pkl
    """
    class_name = object_to_dump.__class__.__name__
    file = class_name + ".pkl"
    file_path = os.path.join(config.DATA_PATH, file)
    joblib.dump(object_to_dump, file_path)
    logging.info("{} object dumped to disk.".format(class_name))
    logging.info("File = {}".format(file_path))


def persist_load(object_file_to_load):
    """Reload sklearn objects dumped with persist_dump."""
    file_path = os.path.join(config.DATA_PATH, object_file_to_load)
    logging.info("Loading pickled object...")
    loaded_object = joblib.load(file_path)
    class_name = loaded_object.__class__.__name__
    logging.info("{} object loaded from disk.".format(class_name))
    return loaded_object


def append_to_csv_file(csv_file, **kwargs):
    """Append some learning parameters and results to a csv file."""
    file_path = os.path.join(config.DATA_PATH, csv_file)
    with open(file_path, 'a') as f:
        f.write(",".join(kwargs.values()))
        f.write("\n")
