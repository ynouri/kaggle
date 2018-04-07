# Data science project config file
import os

# Project name
PROJECT_NAME = 'kaggle-talkingdata2'

# Paths
DATA_BASE_PATH = os.getenv('DATA_BASE_PATH')
DATA_PATH = DATA_BASE_PATH + PROJECT_NAME + '/'

# Comet
COMET = {
    'api_key': os.getenv('COMET_API_KEY'),
    'project_name': PROJECT_NAME,
    'auto_param_logging': True,
    'auto_metric_logging': False,
    'parse_args': False
}
