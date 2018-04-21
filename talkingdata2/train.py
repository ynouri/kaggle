import comet_ml
from unittest.mock import Mock
# import pandas as pd
# import numpy as np
import logging
from sklearn import linear_model
# from sklearn import model_selection
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


def cli_train(file, enable_comet_ml):
    """Train CLI entry point."""
    # Create Comet experiment
    exp = get_experiment(enable_comet_ml)

    # Load dataset with enriched features
    df = data.load(file)
    feature_names = features.get_all_names()

    # Train the model
    logging.info("Start training model...")
    X = df[feature_names]
    y = df.is_attributed
    logreg = linear_model.LogisticRegression(
        solver='sag',
        verbose=1,
        max_iter=200,
        n_jobs=-1
    )
    logreg.fit(X, y)
    logging.info("Model trained.")

    # Persists parameters to disk
    data.persist_dump(logreg)

    # AUC score
    y_score = logreg.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, y_score)
    logging.info("AUC score = {:0.4f}".format(auc_score))
    exp.log_metric("AUC score", auc_score)
    return auc_score
