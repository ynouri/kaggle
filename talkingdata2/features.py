import numpy as np
import logging
# Project modules
import info
import data


def cli_add_features(file):
    """Entry point for the CLI defined in talkingdata.py."""
    # Load raw dataset
    df = data.load(file)
    # Add features
    add_all(df)
    info.memory(df)
    # Save dataset enriched with features
    data.save_hdf(dataframe=df, original_file=file, suffix='with_features')


def add_all(df):
    """
    Take a dataframe and iteratively adds different new features to it.

    Returns the names of the new columns added to the dataframe.
    """
    logging.info("Start adding features...")
    features = []
    features += add_all_nunique(df)
    features += add_click_time_rank(df)
    logging.info("Features added.")
    return features


def add_all_nunique(df):
    """Add various "number of unique" new features to a datafra e."""
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


def add_nunique_by_ip(df, new_feature, original_feature):
    """
    Add the "number of **unique** X for a given IP address" feature to a df.

    X=clicks, apps, devices, os, or channels. The argument original_feature
    defines what feature X is counted. The name of the new feature has to be
    specified in new_feature.
    """
    nunique = df[['ip', original_feature]].drop_duplicates() \
                                          .groupby('ip') \
                                          .count()
    df[new_feature] = df[['ip']].join(
        other=nunique,
        how='left',
        on='ip',
        rsuffix=new_feature
    )[original_feature].astype(dtype=np.uint16).values
    logging.info("Added feature: {}".format(new_feature))
    return [new_feature]


def add_click_time_rank(df):
    """Add the click time order feature to a dataframe, in percentage."""
    new_feature = 'click_time_rank'
    df[new_feature] = df.groupby('ip').click_time.rank(pct=True).values
    logging.info("Added feature: {}".format(new_feature))
    return [new_feature]
