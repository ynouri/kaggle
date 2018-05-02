# TalkingData2

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## Results

The results of the logistic regression models are analyzed [here](results/logreg_results.ipynb).

## How to use

### Add new features

This will create a new hierarchical data format (.hdf) file, for example `train_with_features.hdf`.

```bash
./talkingdata2.py add-features --file train.csv
./talkingdata2.py add-features --file test.csv
```

### Scale the features

This step is specific to logistic regression model. This will create a new .hdf file with scaled features, named for example `train_with_features_scaled.hdf`.

```bash
./talkingdata2.py scale-features --file train_with_features.hdf
./talkingdata2.py scale-features --file test_with_features.hdf --scaler StandardScaler.pkl
```

### Add dummy features (one hot encoder)

This step is specific to random forest model. This will create a new .hdf file wih dummy features, named for example `train_with_features_dummies.hdf`.

```bash
./talkingdata2.py add-dummies --file train_with_features.hdf
./talkingdata2.py add-dummies --file test_with_features.hdf
```

### Train a logistic regression model
This will generate a coefficients file `LogisticRegression.pkl`. Comet ML can be enabled with the `--enable-comet-ml` argument.
```bash
./talkingdata2.py train --file train_with_features_scaled.hdf --model logreg --enable-comet-ml --n-training 10000000
```

### Loop on different training set sizes
```bash
range20kto10m="20000 50000 100000 200000 500000 1000000 2000000 5000000 10000000"
for n in $range20kto10m
do
./talkingdata2.py train --file train_with_features_scaled.hdf --model logreg --n-training $n
done
```

### Train a random forest model
```bash
./talkingdata2.py train --file train_with_features_dummies.hdf --model randomforest --n-training 10000
```

### Generate predictions on the test data

This step will generate a `prediction.csv` file.

```bash
# For logistic regression:
./talkingdata2.py predict --file test_with_features_scaled.hdf --model LogisticRegression.pkl

# For random forest:
./talkingdata2.py predict --file test_with_features_dummies.hdf --model RandomForestClassifier.pkl
```

### Submit predictions using Kaggle API
```bash
prediction=~/data/kaggle-talkingdata2/prediction.csv
kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f $prediction -m 'Submission'
```


## TO DO list, enhancements, questions
* Currently, no prediction can be done on the test dataset because we're using pandas get_dummies function independently on train and test datasets. Since the number of categories is different in both datasets, the classifier doesn't work. We need to move to Scikit-learn OneHotEncoder. Maybe use the sparse format for memory gains. But when to do it? Just before training/prediction? Or can the sparse format be stored on disk with the dataframe?
* Try other classifiers: ~decision trees~, ~random forests~, gradient boosting
* Question (ask to practitioners/Stack overflow): how to handle categorical featured with a lot of categories, on a lot of examples? (e.g. 168M x 400 categories = 60GB, using uint8) Can we gain a x8 factor in memory/disk by stacking bools together? Is this the purpose of using Compressed Sparse Row Format? Is the one hot encoding/dummy computation done before training/prediction? Can the most important categories be pre-defined using for example a PCA?


## DONE list
* ~Submit a first prediction with the logistic regression model, using the Kaggle API.~
* ~Implement new methods but make sure the previous method (logistic regression) can still be used.~
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
