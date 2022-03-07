# -*- coding: utf-8 -*-

import logging

# Default logging level of deep_folding
LOGGING_LEVEL = logging.INFO
LOG_FORMAT = "%(levelname)s:%(name)s: %(message)s"

# Sets up the default logger
logging.basicConfig(level=LOGGING_LEVEL)
ch = logging.StreamHandler()
formatter = logging.Formatter(LOG_FORMAT)
log_module = logging.getLogger('')
log_module.handlers[0].setFormatter(formatter)
