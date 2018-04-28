import train
import features
import predict
import pytest


INPUT = [
    'train_sample.csv',
    'test_sample.csv'
]

WITH_FEATURES = [
    ('train_sample_with_features.hdf', None),
    ('test_sample_with_features.hdf', 'StandardScaler.pkl')
]

SCALED = [
    ('train_sample_with_features_scaled.hdf', None),
    ('test_sample_with_features_scaled.hdf', 'StandardScaler.pkl')
]

LOGREG_MODEL_FILE = "LogisticRegression.pkl"


@pytest.mark.parametrize('file', INPUT)
def test_add_features(file):
    """Test add features."""
    features.cli_add_features(file)


@pytest.mark.parametrize('dataset_file,scaler_file', WITH_FEATURES)
def test_scale_feature(dataset_file, scaler_file):
    """Test scale features."""
    features.cli_scale_features(dataset_file, scaler_file)


def test_train():
    """Test training."""
    file = SCALED[0][0]
    enable_comet_ml = False
    auc_score = train.cli_train(file, enable_comet_ml, 0.60)
    assert auc_score > 0.778


def test_predict():
    """Test predict."""
    predict.cli_predict(SCALED[1][0], LOGREG_MODEL_FILE)
