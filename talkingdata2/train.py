import comet_ml
from unittest.mock import Mock
# import pandas as pd
# import numpy as np
import logging
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# Project modules
import config
import features
import data


def get_experiment(enable_comet_ml=True):
    """Return a comet_ml experiment or a mock object. Useful for unit tests."""
    if enable_comet_ml:
        return comet_ml.Experiment(**config.COMET)
    else:
        return Mock(spec=comet_ml.Experiment)


def train_logreg(X, y):
    """Train a logistic regression model."""
    logging.info("Start training model...")
    logreg = linear_model.LogisticRegression(
        solver='sag',
        verbose=0,
        max_iter=200,
        n_jobs=-1
    )
    logreg.fit(X, y)
    logging.info("Model trained.")
    return logreg


def train_test_split_df(df, n_training):
    """Split the full dataset (with features, scaled) into training and CV."""
    feature_names = features.get_all_names()
    logging.info("Splitting into train and cross-validation sets...")
    X_train, X_cv, y_train, y_cv = train_test_split(
        df[feature_names],
        df.is_attributed,
        train_size=n_training,
        random_state=123,
        shuffle=True
    )
    logging.info("Split done.")
    logging.info("Number of samples:")
    logging.info("\tTraining set = {}".format(len(y_train)))
    logging.info("\tCross validation set = {}".format(len(y_cv)))
    return X_train, X_cv, y_train, y_cv


def cli_train(file, enable_comet_ml, n_training):
    """Train CLI entry point."""
    # Create Comet experiment
    exp = get_experiment(enable_comet_ml)
    exp.log_metric("n_training", n_training)

    # Load dataset with enriched features
    df = data.load(file)

    # Split dataset into train and cross-validation sets
    X_train, X_cv, y_train, y_cv = train_test_split_df(df, n_training)

    # Train the model on training set
    logreg = train_logreg(X_train, y_train)

    # Persists parameters to disk
    data.persist_dump(logreg)

    # AUC score on cross-validation set
    logging.info("Computing AUC score on cross-validation set...")
    y_cv_proba = logreg.predict_proba(X_cv)[:, 1]
    auc_score = roc_auc_score(y_cv, y_cv_proba)
    logging.info("AUC score = {:0.4f}".format(auc_score))
    exp.log_metric("AUC score", auc_score)
    return auc_score
