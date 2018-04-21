import features
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


@pytest.mark.parametrize('file', INPUT)
def test_add_features(file):
    """Test add features."""
    features.cli_add_features(file)


@pytest.mark.parametrize('dataset_file,scaler_file', WITH_FEATURES)
def test_scale_feature(dataset_file, scaler_file):
    """Test scale features."""
    features.cli_scale_features(dataset_file, scaler_file)
