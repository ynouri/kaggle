import pandas as pd
import logging
import data


def cli_predict(features_file, model_file):
    """Prediction CLI entry point."""
    X_predict = data.load(features_file)
    features_no_click_id = [c for c in X_predict.columns if c != 'click_id']
    model = data.persist_load(model_file)
    logging.info("Computing the predicted probabilities...")
    y_predict = model.predict_proba(X_predict[features_no_click_id])[:, 1]
    logging.info("Creating dataframe...")
    df_predict = pd.DataFrame(
        data=y_predict,
        columns=['is_attributed'],
        index=X_predict.click_id
    )
    df_predict.index.name = 'click_id'
    logging.info("Prediction complete.")
    data.save_csv(df_predict, 'prediction.csv')
