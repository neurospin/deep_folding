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
The aim of this script is to compute transformation files from native MRI space
to normalized SPM space

It scans all subjects from src_dir and looks for morphologist analysis folder,
then produces the transformation file from native space to normalized SPM space,
saves the results (one transformation file per subject) into directory tgt_dir

Examples:
    Specifies the source directory where the MRI data lies, the target directory
    where to put the transform, and the number of subjects to analyse
    (all by default)
        $ python transform.py -s /neurospin/hcp -t /path/to/transfo_dir -n all
        $ python transform.py --help
"""

from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import listdir
from os.path import join
from datetime import datetime

import json
import six
import git
from soma import aims

_ALL_SUBJECTS = -1

_SRC_DIR_DEFAULT = "/neurospin/hcp"
_TGT_DIR_DEFAULT = "/neurospin/dico/deep_folding_data/data/transfo_to_spm"

class TransformToSPM:
    """Computes transformation from native to normalized SPM space

    Attributes:
        src_dir: A string giving the name of the source data directory.
        tgt_dir: A string giving the name of the target directory
                in which the transformations are saved.
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT, tgt_dir=_TGT_DIR_DEFAULT):
        """Inits Transform class with source and target directory names

        It also creates the target directory if it doesn't exist

        Args:
            src_dir: string naming src directory
            tgt_dir: string naming target directory in which transformtion files
                     are saved
        """
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir

        # Below are subdirectories and files from the morphologist pipeline
        # Once the database directory (like /neurospin/hcp) is defined,
        # subdirectories remain identical and are specific to the morphologist
        # pipeline
        # 'subject' is the ID of the subject

        # Morphologist directory
        self.morphologist_dir = join(self.src_dir, "ANALYSIS/3T_morphologist")
        # default acquisition subdirectory
        self.acquisition_dir = "%(subject)s/t1mri/default_acquisition"
        # (input) name of normalized SPM file
        self.normalized_spm_file = "normalized_SPM_%(subject)s.nii"
        # (input) name of the raw to MNT Talairach transformation file
        self.to_talairach_file = "registration/" + \
            "RawT1-%(subject)s_default_acquisition_TO_Talairach-MNI.trm"

        # (Outputs) Name of transformation files that are written by the program
        # 'subject' is the ID of the subject
        self.natif_to_spm_file = "natif_to_template_spm_%(subject)s.trm"

    def calculate_one_transform(self, subject_id):
        """Calculates the transformation file of a given subject.

        This transformation enables to go from native space (= MRI space) to
        normalized SPM space. The normalized SPM space (template SPM space) is
        a translation + an axis inversion of the Talairach MNI space.
        The transformation is directly written to the file

        Args:
            subject_id: id of subject of whom transformation file is computed
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        subject = {'subject': subject_id}

        # Names directory where subject analysis files are stored
        subject_dir = \
            join(self.morphologist_dir, self.acquisition_dir % subject)

        # Reads transformation file that goes from native space to
        # Talairach MNI space.
        # The Talairach MNI space has the brain centered, which means
        # that the coordinates can be negative.
        # The normalized SPM (or template SPM) has only positive coordinates
        # and its axes are inverted with respect to Talairach MNI
        to_talairach_file = join(subject_dir, self.to_talairach_file % subject)
        natif_to_mni = aims.read(to_talairach_file)

        # Fetches template's transformation from Talairach MNI to normalized SPM
        # The first transformation[0] of the file normalized_spm
        # is the transformation from normalized SPM to Talairach MNI
        # The transformation between normalized SPM and Talairach MNI
        template = aims.read(
            join(subject_dir, self.normalized_spm_file % subject))
        template_transform = template.header()['transformations'][0]
        mni_to_template = \
            aims.AffineTransformation3d(template_transform).inverse()
        # print(template.header()['transformations'][0])

        # Combination of transformations
        natif_to_template_mni = mni_to_template * natif_to_mni
        # print(natif_to_template_mni)

        # Saving of transformation files
        natif_to_spm_file = join(
            self.tgt_dir, self.natif_to_spm_file % subject)
        aims.write(natif_to_template_mni, natif_to_spm_file)

    def calculate_transforms(self, number_subjects=_ALL_SUBJECTS):
        """Calculates transformation file for all subjects.

        This transformation enables to go from native space
        (= MRI subject space) to normalized SPM space.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if number_subjects:
            # subjects are detected as the directory names under src_dir
            list_all_subjects = listdir(self.morphologist_dir)

            # Gives the possibility to list only the first number_subjects
            list_subjects = (
                list_all_subjects
                if number_subjects == _ALL_SUBJECTS
                else list_all_subjects[:number_subjects])

            # Creates target dir if it doesn't exist
            if not os.path.exists(self.tgt_dir):
                os.mkdir(self.tgt_dir)

            # Creates README
            self.write_readme()

            # Computes and saves transformation files for all listed subjects
            for subject in list_subjects:
                print("subject : " + subject)
                self.calculate_one_transform(subject)


    def write_readme(self):
        """Writes README on the target directory

        It contains information about generation date, git hash/version number,
        source directory and number of subjects
        """

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d -- %H:%M:%S")

        readme_name = join(self.tgt_dir, 'README')
        readme = open(readme_name, 'w')
        readme.write("Current time YYYY-MM-DD = " + current_time + '\n')

        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
            readme.write("To recover the source code used, do:" + '\n')
            readme.write("git checkout " + sha + '\n\n')
            readme.write("repo working dir: " + repo.working_tree_dir + '\n')
        except git.InvalidGitRepositoryError:
            readme.write("No git repository")

        b = self.__dict__
        readme.write('\n' + "Variables internal to TransformToSPM class:\n")
        readme.write(json.dumps(b, sort_keys=False, indent=4))
        readme.close()

def transform_to_spm(src_dir=_SRC_DIR_DEFAULT,
                     tgt_dir=_TGT_DIR_DEFAULT,
                     number_subjects=_ALL_SUBJECTS):
    """High-level API function performing the transform

    Args:
        src_dir: source directory name, full path
        tgt_dir: target directory where to save the transformations, full path
        number_subjects: number of subjects to analyze (all=-1 by default)
    """

    # Do the actual transformations
    transformer = TransformToSPM(src_dir=src_dir, tgt_dir=tgt_dir)
    transformer.calculate_transforms(number_subjects=number_subjects)


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        src_dir: source directory name, full path
        tgt_dir: target directory where to save the transformations, full path
        number_subjects: number of subjects to analyze
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='transform.py',
        description='Generate transformation files')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the MRI data lies. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the output transformation files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'.'
             '0 subject is allowed, for debug purpose.'
             'Default is : all')

    args = parser.parse_args(argv)
    src_dir = args.src_dir
    tgt_dir = args.tgt_dir
    number_subjects = args.nb_subjects

    # Check if nb_subjects is either the string "all" or a positive integer
    try:
        if number_subjects == "all":
            number_subjects = _ALL_SUBJECTS
        else:
            number_subjects = int(number_subjects)
            if number_subjects < 0:
                raise ValueError
    except ValueError:
        raise ValueError(
            "nb_subjects must be either the string \"all\" or an integer")

    return src_dir, tgt_dir, number_subjects


def main(argv):
    """Reads argument line and creates transformation files

    These are transformations from native to normalize SPM space

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        src_dir, tgt_dir, number_subjects = parse_args(argv)
        # Actual API
        transform_to_spm(src_dir, tgt_dir, number_subjects)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # We do this to be able to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
