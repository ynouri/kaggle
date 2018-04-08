# TalkingData2

https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

## Train and evaluate the model on AWS

```bash
# Export DATA_BASE_PATH and COMET_API_KEY env variables
# (if not already done by default in your environment)
vim .env
source .env

# Run on the sample or full training set
./run.py training_sample.csv
```
