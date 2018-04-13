import numpy as np
import pandas as pd
import logging
import sys
# Project modules
import info
import load

# CLI entry point

def cli_add_features(file):

    # Load data
    df = load.data(file)

    # Add features
    logging.info("Start adding features...")
    feature_labels = add_all(df)
    logging.info("Features added.")
    info.memory(df)

# Add all features function

def add_all(df):
    features = []
    features += add_all_nunique(df)
    features += add_click_time_rank(df)
    return features

# Number of unique X (where X=clicks, apps, devices, os, channels) for a given IP

def add_nunique_by_ip(df, new_feature, original_feature):
    nunique = df[['ip', original_feature]].drop_duplicates() \
                                          .groupby('ip') \
                                          .count()
    df[new_feature] = df[['ip']].join(
        other=nunique,
        how='left',
        on='ip',
        rsuffix=new_feature
    )[original_feature].values
    logging.info("Added feature: {}".format(new_feature))
    return [new_feature]

def add_all_nunique(df):
    features_nunique = {
        'total_clicks_by_ip': 'click_time',
        'nunique_apps_by_ip': 'app',
        'nunique_devices_by_ip': 'device',
        'nunique_os_by_ip': 'os',
        'nunique_channels_by_ip': 'channel',
    }
    for (new_feature, original_feature) in features_nunique.items():
        add_nunique_by_ip(df, new_feature, original_feature)
    return features_nunique.keys()


# Rank of the click time, in percentage

def add_click_time_rank(df):
    new_feature = 'click_time_rank'
    df[new_feature] = df.groupby('ip').click_time.rank(pct=True).values
    logging.info("Added feature: {}".format(new_feature))
    return [new_feature]
