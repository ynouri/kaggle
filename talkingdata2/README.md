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
./talkingdata2.py scale-features --file train_with_features.hdf
./talkingdata2.py scale-features --file test_with_features.hdf --scaler StandardScaler.pkl

# Train the linear regression model
# It will generate coefficients file linreg_coeffs.csv
./talkingdata2.py train --file train_with_features_scaled.hdf --enable-comet-ml --n-training 10000

# Generate predictions on the test data
# A file predictions.csv is generated
./talkingdata2.py predict --file test_with_features.hdf --coeffs linreg_coeffs.csv
```

## TO DO list & enhancement ideas
* Compute performance of scikit-learn logistic regression for different training set sizes (n=10k, 100k, 1m, 10m) and infer total calibration time needed on 1 CPU
* Investigate multi-threading or multi-worker calibration and what that would mean in terms of modeling (averaging of parameters at job manager level??)
* Try different solvers outside SAG: e.g. SAGA
* ~~Convert initial CSV file to HDF to get a faster loading~~
* ~~Memory usage: scaled features data type can be a float32 or even a float16, probably wouldn't cost too much precision.~~
* ~~Memory usage: when the features used for model training have been enriched, the other features are not needed anymore, they can be dropped to save memory usage and disk space (hdf file will be smaller)~~
* ~~Save intermediary data results (in hdf5 format) to S3~~
* ~~Features preparation process has to be streamlined so that it can be applied directly to various input files (sample, training, supplement, test)~~
* ~~Features preparation can live in a separate script.~~
* ~~Scikit-learn logistic regression should be used with a SAG or SAGA optimizer for better results --> need to inspect the performance in a notebook.~~
* ~~When modularizing the project, remember that different models attempts will have to co-exist and be compared, potentially blended~~
