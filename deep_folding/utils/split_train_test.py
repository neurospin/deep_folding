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

""" Splits the subjects IDs in a train set and a test set.

Each set is represented as a csv file (train.csv and test.csv).
The first column on each row is the subject ID.
"""

import argparse
import os
import random
import csv
import sys
import six

_SRC_DIR_DEFAULT = "/host/tgcc/hcp/ANALYSIS/3T_morphologist"
_TGT_DIR_DEFAULT = "."
_NB_TEST_SUBJECTS_DEFAULT = 150
_SEED = 2

def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args: Namespace
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='split_train_test.py',
        description='Splits subject IDs in train and test csv files')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the subjects lie as subdirectory. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the csv files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-n", "--nb_test_subjects", type=int, default=_NB_TEST_SUBJECTS_DEFAULT,
        help='Number of subjects of the test set')

    return parser.parse_args(argv)

def split_train_test(src_dir, tgt_dir, nb_test_subjects):
    """Splits the subject IDs in train and test
    """

    # Lists all subjects
    subjects = os.listdir(src_dir)

    # Makes the random split on list

    random.seed(_SEED)
    random.shuffle(subjects)

    test_subjects = subjects[:nb_test_subjects]
    train_subjects = subjects[nb_test_subjects:]

    # Saves subject IDs is csv files

    train_filename = f'{tgt_dir}/train.csv'
    with open(train_filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for subject in train_subjects:
            wr.writerow([subject])

    test_filename = f'{tgt_dir}/test.csv'
    with open(test_filename, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for subject in test_subjects:
            wr.writerow([subject])



def main(argv):
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        args = parse_args(argv)
        # Actual API
        split_train_test(args.src_dir, args.tgt_dir, args.nb_test_subjects)

    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())

######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
