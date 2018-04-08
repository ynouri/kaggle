import logging

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

# Memory usage
def memory(df):
    mem_gb = df.memory_usage().sum() / 1024 ** 3
    logging.info("Dataframe memory usage = {:0.2f}GB".format(mem_gb))

# Number of rows
def rows(df):
    logging.info("Number of rows = {:,}".format(len(df)))
