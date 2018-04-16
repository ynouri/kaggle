# TalkingData2

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## How to use

```bash
# Add the features to the model
# It will generate train_with_features.hdf.
# Do it both for train & test sets
./talkingdata2.py add-features --file train.csv
./talkingdata2.py add-features --file test.csv

# Scale the features
./talkingdata2.py scaler-features --file train_with_features.hdf
./talkingdata2.py scaler-features --file test_with_features.hdf --scaler StandardScaler.pkl

# Train the linear regression model
# It will generate coefficients file linreg_coeffs.csv
./talkingdata2.py train --file train_with_features.hdf

# Generate predictions on the test data
# A file predictions.csv is generated
./talkingdata2.py predict --file test_with_features.hdf --coeffs linreg_coeffs.csv
```

## TO DO list & enhancement ideas
* Convert initial CSV file to HDF to get a faster loading
* The new features datatypes need to be optimized so that they can take less memory space (adding 6 features took the df object from 4.99GB to 13.26GB)
* Save intermediary data results (in hdf5 format) to S3
* Features preparation process has to be streamlined so that it can be applied directly to various input files (sample, training, supplement, test)
* Features preparation can live in a separate script.
* Scikit-learn logistic regression should be used with a SAG or SAGA optimizer for better results --> need to inspect the performance in a notebook.
* When modularizing the project, remember that different models attempts will have to co-exist and be compared, potentially blended


## Train and evaluate the model on AWS

```bash
# Export DATA_BASE_PATH and COMET_API_KEY env variables
# (if not already done by default in your environment)
vim .env
source .env

# Run on the sample or full training set
./run.py training_sample.csv
```
