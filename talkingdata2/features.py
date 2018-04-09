import numpy as np
import pandas as pd
import logging

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


# Add all features function

def add_all(df):
    features = []
    features += add_all_nunique(df)
    features += add_click_time_rank(df)
    return features
