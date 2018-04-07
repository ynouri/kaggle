# TalkingData2

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## Train and evaluate the model on AWS

```bash
# Start a beefy instance

# Sync the data already stored on S3
aws s3 sync s3://nouri-bucket/data/kaggle-talkingdata2 ~/data/kaggle-talkingdata2

# Download the code from github
git clone https://github.com/ynouri/kaggle.git

# Export DATA_BASE_PATH and COMET_API_KEY env variables
# (if not already done by default in your environment)
vim .env
source .env

# Run on the sample or full training set
./run.py training_sample.csv
```
