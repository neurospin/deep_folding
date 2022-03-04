# -*- coding: utf-8 -*-

import logging

# Default logging level of deep_folding
LOGGING_LEVEL = logging.INFO

# Sets up the default logger
logging.basicConfig(level=LOGGING_LEVEL)
log_module = logging.getLogger()