import train
import features
import predict
import pytest


INPUT = [
    'train_sample.csv',
    'test_sample.csv'
]

WITH_FEATURES = [
    'train_sample_with_features.hdf',
    'test_sample_with_features.hdf',
]

WITH_DUMMIES = [
    'train_sample_with_features_dummies.hdf',
    'test_sample_with_features_dummies.hdf'
]

MODEL_FILE = 'RandomForestClassifier.pkl'


# TODO: refactorize this test with test_run_logreg.py and re-use the same file.
@pytest.mark.parametrize('file', INPUT)
def test_add_features(file):
    """Test add features."""
    features.cli_add_features(file)


@pytest.mark.parametrize('dataset_file', WITH_FEATURES)
def test_add_dummies(dataset_file):
    """Test add dummy features."""
    features.cli_add_dummies(dataset_file)


def test_train():
    """Test training."""
    file = WITH_DUMMIES[0]
    enable_comet_ml = False
    model = 'randomforest'
    auc_score = train.cli_train(file, model, enable_comet_ml, 0.60)
    assert auc_score > 0.92


def test_predict():
    """Test predict."""
    predict.cli_predict(WITH_DUMMIES[1], MODEL_FILE)
