#!/usr/bin/env python3
import click
# Project modules
import features
import train
import predict

# Add features sub-command
@click.command(name='add-features')
@click.option('--file', default='train_sample.csv',
              help='Training data CSV or HDF file')
def add_features(file):
    features.cli_add_features(file)

# Train model sub-command
@click.option('--file', help='Feature enriched dataset HDF file')
@click.command(name='train')
def train_():
    train.cli_train(file)

# Predict sub-command
@click.option('--file', help='Feature enriched dataset HDF file')
@click.option('--parameters', help='Trained model parameters')
@click.command(name='predict')
def predict_():
    predict.cli_predict(file, parameters)

# Main entry point, groups several sub-commands
@click.group()
def main():
    pass

main.add_command(add_features)
main.add_command(train_)
main.add_command(predict_)

if __name__ == "__main__":
    main()
