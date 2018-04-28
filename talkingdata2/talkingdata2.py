#!/usr/bin/env python3
import click
# Project modules
import train
import features
import predict


@click.command(name='add-features')
@click.option('--file', default='train_sample.csv',
              help='Training data CSV or HDF file')
def add_features(file):
    """Add features to the dataset."""
    features.cli_add_features(file)


@click.command(name='scale-features')
@click.option('--file', default='train_sample_with_features.hdf',
              help='Training data CSV or HDF file, enriched with features.')
@click.option('--scaler', default=None,
              help='StandardScaler joblib dump file.')
def scale_features(file, scaler):
    """Scale dataset features."""
    features.cli_scale_features(file, scaler)


@click.command(name='train')
@click.option('--file', default='train_sample_with_features.hdf',
              help='Feature enriched dataset HDF file')
@click.option('--enable-comet-ml', is_flag=True,
              help='Enable experiment results collection with Comet ML.')
@click.option('--n-training', default=0.60,
              help='Training set size, can be a float or an integer.')
def train_(file, enable_comet_ml, n_training):
    """Train the model."""
    if n_training >= 1.0:
        n_training = int(n_training)
    train.cli_train(file, enable_comet_ml, n_training)


@click.command(name='predict')
@click.option('--file', help='Feature enriched dataset HDF file')
@click.option('--model', help='Joblib dump (.pkl) of the trained model')
def predict_(file, parameters):
    """Predict based on trained model and generate results to .csv."""
    predict.cli_predict(file, model)


@click.group()
def main():
    """Group the add-features, train and predict sub-commands."""
    pass


main.add_command(add_features)
main.add_command(scale_features)
main.add_command(train_)
main.add_command(predict_)


if __name__ == "__main__":
    main()
