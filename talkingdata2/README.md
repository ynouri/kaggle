# TalkingData2

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## Results

The results of the logistic regression models are analyzed [here](results/logreg_results.ipynb).

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
# It will generate coefficients file LogisticRegression.pkl
./talkingdata2.py train --file train_with_features_scaled.hdf --enable-comet-ml --n-training 10000000
# Without Comet:
./talkingdata2.py train --file train_with_features_scaled.hdf --n-training 10000000

# To loop on different training set sizes:
range20kto1m="20000 50000 100000 200000 500000 1000000"
range2mto50m="2000000 5000000 10000000 20000000 50000000"
for n in $range2mto50m
do
./talkingdata2.py train --file train_with_features_scaled.hdf --n-training $n
done

# Generate predictions on the test data
./talkingdata2.py predict --file test_with_features_scaled.hdf --model LogisticRegression.pkl

# Submit prediction using Kaggle API
prediction=~/data/kaggle-talkingdata2/prediction.csv
kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f $prediction -m 'Submission'
```

## TO DO list & enhancement ideas
* Submit a first prediction with the logistic regression model, using the Kaggle API.
* Try other classifiers: decision trees, random forests, gradient boosting
* Implement new methods but make sure the previous method (logistic regression) can still be used.

## DONE list
* ~~Compute performance of scikit-learn logistic regression for different training set sizes (n=10k, 100k, 1m, 10m) and infer total calibration time needed on 1 CPU~~
* ~~Investigate multi-threading or multi-worker calibration and what that would mean in terms of modeling (averaging of parameters at job manager level??)~~
* ~~Try different solvers outside SAG: e.g. SAGA~~
* ~~Convert initial CSV file to HDF to get a faster loading~~
* ~~Memory usage: scaled features data type can be a float32 or even a float16, probably wouldn't cost too much precision.~~
* ~~Memory usage: when the features used for model training have been enriched, the other features are not needed anymore, they can be dropped to save memory usage and disk space (hdf file will be smaller)~~
* ~~Save intermediary data results (in hdf5 format) to S3~~
* ~~Features preparation process has to be streamlined so that it can be applied directly to various input files (sample, training, supplement, test)~~
* ~~Features preparation can live in a separate script.~~
* ~~Scikit-learn logistic regression should be used with a SAG or SAGA optimizer for better results --> need to inspect the performance in a notebook.~~
* ~~When modularizing the project, remember that different models attempts will have to co-exist and be compared, potentially blended~~
