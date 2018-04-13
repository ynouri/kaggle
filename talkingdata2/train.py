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
import data


# CLI entry point

def cli_train(file):

    # Create Comet experiment
    exp = Experiment(**config.COMET)

    # Train the model
    logging.info("Start training model...")
    X = df[feature_labels]
    y = df.is_attributed
    logreg = linear_model.LogisticRegression()
    logreg.fit(X, y)
    logging.info("Model trained.")

    # AUC score
    y_score = logreg.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_score)
    logging.info("AUC score = {:0.4f}".format(auc_score))
    exp.log_metric("AUC score", auc_score)
