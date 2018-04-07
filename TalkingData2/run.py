#!/usr/bin/env python3
import sys
from comet_ml import Experiment
import pandas as pd
import numpy as np
import logging
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# Project modules
import config
import features
import info

# Main script
if __name__ == '__main__':

    # Comet experiment
    exp = Experiment(**config.COMET)

    # Load data
    file = sys.argv[1]
    file_path = config.DATA_PATH + file
    exp.log_parameter('File', file)
    logging.info("File = {}".format(file_path))
    logging.info("Loading dataframe...")
    df = pd.read_csv(file_path)
    logging.info("Dataframe loaded.")
    info.rows(df)
    info.memory(df)

    # Add features
    logging.info("Start adding features...")
    feature_labels = features.add_all(df)
    info.memory(df)

    # Train the model
    X = df[feature_labels]
    y = df.is_attributed
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)

    # AUC score
    y_score = logreg.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_score)
    logging.info("AUC score = {:0.4f}".format(auc_score))
    exp.log_metric("AUC score", auc_score)
