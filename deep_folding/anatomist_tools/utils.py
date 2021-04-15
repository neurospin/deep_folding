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
used by brainvisa-dependent preprocessing
"""

import json
import os
import errno
import time
import git
from datetime import datetime


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
            print("File " + self.json_file + " cannot be overwritten")

    def update(self, dict_to_add):
        """Updates json file with new dictionary entry

        Args:
            dict_to_add: dictionary appended to json file
        """

        try:
            with open(self.json_file, "r") as json_file:
                data = json.load(json_file)
        except IOError:
            print("File %s is not readable through json.load", self.json_file)

        data.update(dict_to_add)

        try:
            with open(self.json_file, "w") as json_file:
                json_file.write(json.dumps(data, sort_keys=False, indent=4))
        except IOError:
            print("File %s is not writable", self.json_file)

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
