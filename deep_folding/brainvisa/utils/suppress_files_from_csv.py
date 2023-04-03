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
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the

"""
This program converts volumes contained in a folder into buckets.
It writes bucket files in the output folder
"""
import argparse
import sys
import os
import csv
import six


def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='suppress_files_from_csv.py',
        description='Suppress files listed in csv')
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True,
        help='csv file containing file names to suppress.')

    args = parser.parse_args(argv)

    return args


def suppress(csv_file_name):
    """Suppress files listed in csv
    """
    print(csv_file_name)
    print(f"Suppressing filenames contained in {csv_file_name}...", end='')
    removed = 0
    with open(csv_file_name, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            filename = row[0]
            print(".", end='')
            if os.path.isfile(filename):
                removed += 1
                print(filename)
                os.remove(filename)
                os.remove(f"{row[0]}.minf")
    print("DONE")
    print(f"Number of removed files = {removed}")
    print(f"Number of files in csv = {idx+1}")


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
        suppress(args.csv_file)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
