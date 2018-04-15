import logging


# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S'
)


def memory(df):
    """Log at info level the memory usage of a dataframe."""
    mem_gb = df.memory_usage().sum() / 1024 ** 3
    logging.info("Dataframe memory usage = {:0.2f}GB".format(mem_gb))


def rows(df):
    """Log at info level the number of rows in a dataframe."""
    logging.info("Number of rows = {:,}".format(len(df)))
