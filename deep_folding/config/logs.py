# -*- coding: utf-8 -*-

"""It provides log configuration and utilities"""

import logging
from os.path import basename
from stat import filemode

# Default logging level of deep_folding
LOGGING_LEVEL = logging.INFO
LOG_FORMAT = "%(levelname)s:%(name)s: %(message)s"

# Sets up the default logger
logging.basicConfig(
    level=LOGGING_LEVEL,
    handlers=[
        logging.StreamHandler()
    ])
ch = logging.StreamHandler()
formatter = logging.Formatter(LOG_FORMAT)
log_deep_folding = logging.getLogger('')
for hdlr in log_deep_folding.handlers[:]:
    hdlr.setFormatter(formatter)


def set_root_logger_level(verbose_level):
    """Sets root logger level
    
    if verbose_level is:
        - 0: logging.WARNING is selected
        - 1: logging.INFO is selected
        - >1: logging.DEBUG is selected

    Args:
        verbose_level: int giving verbose level"""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose_level, len(levels) - 1)] # cap to last level
    root = logging.getLogger()
    root.setLevel(level)


def set_file_logger(path_file):
    """Returns specific file logger
    
    Args:
        path_file: string giving file name with path (__file__)

    Returns: file-specific logger
    """
    global log_deep_folding
    return log_deep_folding.getChild(basename(path_file))


def set_file_log_handler(file_dir, suffix):
    """Sets file handler for all logs.
    
    Args:
        file_dir: string with folder for file log
        suffix: string -> name of log file = log_{suffix}.log
    """
    global log_deep_folding
    global formatter

    # Creates filename
    suffix = suffix.rstrip('.')
    file_name = f"{file_dir}/log_{suffix}.log"

    # Creates handler
    filehandler = logging.FileHandler(file_name, mode='w')

    # Substitutes file handler in main logger
    for hdlr in log_deep_folding.handlers[:]:
        if isinstance(hdlr, logging.FileHandler):
            log_deep_folding.removeHandler(hdlr)
    log_deep_folding.addHandler(filehandler)
    for hdlr in log_deep_folding.handlers[:]:
        hdlr.setFormatter(formatter)

    # Logs name of log fle
    simple_critical_log(log=log_deep_folding,
                        log_message=f"\nLog written to:\n{file_name}\n")


def simple_critical_log(log, log_message):
    """Prints simple log with only message printed out
    
    Args:
        log: logger
        log_message: string being log message to be printed
    """
    global log_deep_folding

    old_format = []
    for hdlr in log_deep_folding.handlers[:]:
        old_format.append(hdlr.formatter)
        new_formatter = logging.Formatter('%(message)s')
        hdlr.setFormatter(new_formatter)
    log.critical(log_message)
    for hdlr, form in zip(log_deep_folding.handlers[:], old_format):
        hdlr.setFormatter(form)
