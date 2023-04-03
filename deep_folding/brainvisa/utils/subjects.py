#!python
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

import os
import glob
import re

import pandas as pd

from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

from deep_folding.config.logs import set_file_logger
# Defines logger
log = set_file_logger(__file__)


def get_number_subjects(nb_subjects: str) -> int:
    """Returns nb_subjects as int

    If it is \"all\", it returns -1

    Args:
        nb_subjects: string giving nb of subjects (\"all\" if all subjects)

    Returns:
        nb_subjects_int: int giving nb of subjects (-1 if all subjects)
    """
    try:
        if nb_subjects == "all":
            nb_subjects_int = _ALL_SUBJECTS
        else:
            nb_subjects_int = int(nb_subjects)
            if nb_subjects_int < 0:
                raise ValueError
    except ValueError:
        raise ValueError(
            "number_subjects must be either the string \"all\" or an integer")
    return nb_subjects_int


def select_subjects_int(orig_list: list, nb_subjects: int) -> list:
    """Returns a sublist of nb_subjects elements

    if nb_subjects == -1, it returns the original list

    Args:
        orig_list: list of strings, the origin list of subjects
        nb_subjects: intgiving nb of subjects (-1 if all subjects)

    Returns:
        sublist: list of strings, being the select number of subjects
    """
    sublist = (
        orig_list
        if nb_subjects == _ALL_SUBJECTS
        else orig_list[:nb_subjects])

    return sublist


def select_subjects(orig_list: list, nb_subjects: str) -> list:
    """Returns a sublist of nb_subjects elements

    if nb_subjects == \"all\", it returns the original list
    Otherwise it returns the nb_subjects first elements of orig_list

    Args:
        orig_list: list of strings, the origin list of subjects
        nb_subjects: string giving nb of subjects ("all" if all subjects)

    Returns:
        sublist: list of strings, being the select number of subjects
    """
    nb_subjects_int = get_number_subjects(nb_subjects)
    sublist = select_subjects_int(orig_list, nb_subjects_int)

    return sublist


def get_all_subjects_as_dictionary(src_dir_list,
                                   graph_file_list,
                                   side):
    """Lists all subjects from the database (directory src_dir).

    Subjects are the names of the subdirectories of the root directory.

    Returns:
        subjects: a list of dictionaries containing all subjects as dict
    """

    subjects = []

    # Main loop: list all subjects of the directories
    # listed in self.src_dir
    for src_dir, graph_file in zip(src_dir_list, graph_file_list):
        list_src_dir = os.listdir(src_dir)
        if len(list_src_dir) == 0:
            raise RuntimeError(f"source directory {src_dir} is empty!")
        for filename in list_src_dir:
            directory = os.path.join(src_dir, filename)
            if os.path.isdir(directory):
                if filename != 'ra':
                    subject = filename
                    subject_d = {
                        'subject': subject,
                        'side': side,
                        'dir': src_dir,
                        'graph_file': graph_file % {
                            'side': side,
                            'subject': subject}}
                    subjects.append(subject_d)

    log.info(f"Number of subjects in directories: {len(subjects)}\n")

    return subjects


def select_good_qc(orig_list: list, qc_path: str):
    """Return the sublist of orig_list where all data with bad qc are removed.
    /!\\ Also removes subjects that are not mentioned in the qc file.

    Args:
        orig_list: list of strings, the origin list of subjects
        qc_path: string giving the file path with the relevant information.
            It is  a .csv (or .tsv) with a 'participant_id' and 'qc' columns.
            qc values are supposed to be either 0 or 1.
            If qc_path is set to None, then no QC are applied.

    Returns:
        subjects: list of strings, being the subjects with acceptable qc.
    """
    if qc_path == '':
        # then no QC are applied
        sublist = orig_list

    else:
        log.info(f'Treat quality checks from {qc_path}')
        if '.tsv' in qc_path:
            sep = '\t'
        else:
            sep = ','
        log.info(f'Reading qc tsv file')
        qc_file = pd.read_csv(qc_path, sep=sep)

        qc_file = qc_file[qc_file.qc != 0]

        sublist = [name for name in orig_list
                   if name in qc_file.participant_id.values]

        log.info(
            f"{len(set(orig_list) - set(sublist))} "
            f"subjects have been removed because of the qc. "
            f"They are the following: {set(orig_list) - set(sublist)}")

    return sublist


def is_it_a_subject(filename):
    if re.search('.minf$', filename):
        return False
    elif re.search('.sqlite$', filename):
        return False
    elif re.search('.html$', filename):
        return False
    else:
        return True
