#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

"""
The aim of this script is to put together useful classes and functions
used by brainvisa-dependent preprocessing for logging
"""

import errno
import json
import numbers
import os
import sys
import time
from argparse import Namespace
from datetime import datetime
from venv import create

import git

from deep_folding.config.logs import set_file_logger
from deep_folding.config.logs import log_deep_folding
from deep_folding.config.logs import simple_critical_log
from deep_folding.config.logs import set_root_logger_level
from deep_folding.config.logs import set_file_log_handler
from .folder import create_folder

# Defines logger
log = set_file_logger(__file__)


class LogJson:
    """Handles json file lifecycle

    Json file is created by overwriting old file.
    Upon loading a ne dictionary, it updates the json file
    by reading back and writing the updated content
    """

    def __init__(self, json_file):
        """Creates json file and updates its content

        Args:
            json_file: string giving the path/filename to the json file
        """
        self.json_file = json_file
        self.create_file()

    def create_file(self):
        """Creates json file and overwrites old content
        """

        if not os.path.exists(os.path.dirname(self.json_file)):
            try:
                os.makedirs(os.path.dirname(self.json_file))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        try:
            with open(self.json_file, "w") as json_file:
                json_file.write(json.dumps({}))
        except IOError:
            log.critical("File " + self.json_file + " cannot be overwritten")

    def update(self, dict_to_add):
        """Updates json file with new dictionary entry

        Args:
            dict_to_add: dictionary appended to json file
        """

        try:
            with open(self.json_file, "r") as json_file:
                data = json.load(json_file)
        except IOError:
            log.critical(
                "File %s is not readable through json.load", self.json_file)

        data.update(dict_to_add)

        try:
            with open(self.json_file, "w") as json_file:
                json_file.write(json.dumps(data, sort_keys=True, indent=4))
        except IOError:
            log.critical("File %s is not writable", self.json_file)

    def write_general_info(self):
        """Writes general information on json

        It contains information about generation date, git hash/version number.
        """

        # Writes time on json
        timestamp_now = time.time()
        date_now = datetime.fromtimestamp(timestamp_now)
        dict_to_add = {'timestamp': timestamp_now,
                       'date': date_now.strftime('%Y-%m-%d %H:%M:%S')}

        # Writes git information on dictionary if avauilable
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            dict_to_add.update({'is_git': True,
                                'git_sha': sha,
                                'repo_working_dir': repo.working_tree_dir})
        except git.InvalidGitRepositoryError:
            dict_to_add['is_git'] = False

        # Updates json file with new dictionary by reading and writing the file
        self.update(dict_to_add=dict_to_add)


def log_command_line(args: Namespace,
                     prog_name: str,
                     tgt_dir: str,
                     suffix: str = None) -> None:
    """Logs command on file command_line.sh in target directory

    The command file gives thus the exact command line
    the should be given to reproduce the results"""

    global log_deep_folding

    # Builds the effective command line
    log.debug(f"type of args = {type(args)}")
    log.debug(f"args = {args}")
    cmd_line = f"python3 {prog_name}"
    args_dict = vars(args)
    log.debug(f"args_dict = {args_dict}")
    for key in args_dict:
        if isinstance(args_dict[key], bool):
            if args_dict[key]:
                cmd_line += " --" + key
        elif isinstance(args_dict[key], list):
            cmd_line += " --" + key + " " \
                        + ' '.join([str(e) for e in args_dict[key]])
        elif key == "verbose":
            for _ in range(args_dict[key]):
                cmd_line += " --verbose"
        elif isinstance(args_dict[key], numbers.Number):
            cmd_line += " --" + key + " " + str(args_dict[key])
        elif args_dict[key]:
            cmd_line += " --" + key + " " + args_dict[key]

    simple_critical_log(log=log,
                        log_message=f"\nBash command:\n$ {cmd_line}\n")

    # Builds the name of command line file, which is a bash script file
    create_folder(tgt_dir)
    if suffix:
        suffix = suffix.rstrip('.')
        cmd_line_file = f"{tgt_dir}/command_line_{suffix}.sh"
    else:
        cmd_line_file = f"{tgt_dir}/command_line.sh"

    # Save a reference to the original standard output
    original_stdout = sys.stdout

    # This writes the command line into the command line script file
    with open(cmd_line_file, 'w') as f:
        # Change the standard output to the file we created.
        sys.stdout = f
        print("#!/bin/sh")
        print(cmd_line)

    # Reset the standard output to its original value
    sys.stdout = original_stdout


def setup_log(args: Namespace,
              log_dir: str,
              prog_name: str,
              suffix: str = None) -> None:
    """Setups log for command line programs"""

    # Sets level of root logger
    set_root_logger_level(args.verbose + 1)
    # Sets handler for deep_folding logger
    set_file_log_handler(file_dir=log_dir,
                         suffix=suffix)

    # Writes command line argument to target dir for logging
    log_command_line(args,
                     prog_name=prog_name,
                     tgt_dir=log_dir,
                     suffix=suffix)
