import logging
import os

logger = logging.getLogger("fractaldna")

log_level = os.environ.get("FRACTALDNA_LOG_LEVEL", "ERROR")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(log_level)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to ch
ch.setFormatter(formatter)
