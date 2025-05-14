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

""" Add left and right volumes.
"""

import argparse
import glob
import re
import sys
from os.path import abspath, basename, join, exists, isdir
import numpy as np
from p_tqdm import p_map
from soma import aims

from deep_folding.brainvisa.utils.skeleton import is_skeleton
from deep_folding.brainvisa.utils.subjects import is_it_a_subject
from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected
from deep_folding.config.logs import set_file_logger

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, _SKELETON_DIR_DEFAULT \

_SRC_FILENAME_DEFAULT = "resampled_skeleton"
_OUTPUT_FILENAME_DEFAULT = "resampled_skeleton"

# Defines logger
log = set_file_logger(__file__)

def parse_args(argv):
    """Parses command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description="Generate volumes without ventricle.")
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SKELETON_DIR_DEFAULT,
        help="Directory with the two volumes to be added"
             f"Default is : {_SKELETON_DIR_DEFAULT}")
    parser.add_argument(
        "-f", "--src_filename", type=str,
        default=_SRC_FILENAME_DEFAULT,
        help="Filename of source files. "
        "Format is : <SIDE><source_filename>_<SUBJECT>.nii.gz "
        f"Default is : {_SRC_FILENAME_DEFAULT}")
    parser.add_argument(
        "-e", "--output_filename", type=str, default=_OUTPUT_FILENAME_DEFAULT,
        help="Filename of output files. "
             "Format is : <SIDE><output_filename>_<SUBJECT>.nii.gz "
             f"Default is : {_OUTPUT_FILENAME_DEFAULT}")
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')

    args = parser.parse_args(argv)
    setup_log(args,
              log_dir=f"{args.src_dir}",
              prog_name=basename(__file__),
              suffix="F")

    params = vars(args)
    params['src_dir'] = abspath(args.src_dir)
    # Checks if nb_subjects is either the string "all" or a positive integer
    # params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


class AddLeftandRightVolumes:
    """ Class to add right and left volumes of a subject.

    It contains two sanity checks : one to ensure that left and rigth volumes
    have the same header, and another one to ensure that there are not too much
    voxels with different values in right and left volumees.
    """

    priority_order = {0: 15, 10: 10, 11: 14, 20: 9, 30: 12, 35: 11, 40: 8,
                      50: 7, 60: 13, 70: 6, 80: 5, 90: 4, 100: 3, 110: 2, 120: 1}

    def __init__(self, src_dir, 
                 src_filename, output_filename,
                 parallel):

        self.parallel = parallel
        self.src_dir = src_dir
        self.output_dir = join(self.src_dir, "F")
        create_folder(abspath(self.output_dir))

        self.expr = f"^.{src_filename}_(.*).nii.gz$"
        self.src_file = f"%(side)s{src_filename}_%(subject)s.nii.gz"
        self.output_file = f"F{output_filename}_%(subject)s.nii.gz"

    def add_left_and_right_volumes_from_one_subject(self, subject: str):
        """ Add left and right volumes and writes new volume file for one subject.
        """

        src_file_left = join(self.src_dir, "L", self.src_file % {"subject": subject, "side": "L"})
        src_file_right = join(self.src_dir, "R", self.src_file % {"subject": subject, "side": "R"})
        output_file = join(self.output_dir, self.output_file % {"subject": subject})

        log.debug(f"src_file : {src_file_left} - {src_file_right}")
        log.debug(f"output_file = {output_file}")

        if exists(src_file_left) & exists(src_file_right):
            volume_left = aims.read(src_file_left)
            volume_right = aims.read(src_file_right)

            # Sanity check
            for key in volume_left.header().keys():
                if not volume_left.header()[key] == volume_right.header()[key]:
                    log.warning(f"Left and right volumes have different values in header for {key} : {volume_left.header()[key]} / {volume_right.header()[key]}")
            
            # Add the left and right skeletons
            volume = volume_left + volume_right
            array_left = np.asarray(volume_left)
            array_right = np.asarray(volume_right)
            array = np.asarray(volume)

            # For contentious voxels (voxels which have two different values in the two skeletons),
            # the value is chosen according to the priority order
            vectorize = np.vectorize(lambda x: self.priority_order[x])
            mask = vectorize(array_right) >= vectorize(array_left)
            array[mask] = array_left[mask]
            array[~mask] = array_right[~mask]
            array = array.astype(int)

            # Sanity checks
            # FIXME : select good threshold
            threshold = 200
            nb_contentious_voxels = np.count_nonzero(np.logical_and(array_left, array_right))
            log.debug(f"Unique value of the volume : {np.unique(array)}")
            log.debug(f"Number of conflict voxels between left and right volumes : {nb_contentious_voxels}")
            if nb_contentious_voxels > threshold:
                log.warning(f"Left and right volumes have {nb_contentious_voxels} voxels with different values ! "
                            f"Volume files : {src_file_left} and {src_file_left}")

            if not is_skeleton(array):
                log.warning(f"Volume has unexpected skeleton values: {np.unique(array)}")
            
            aims.write(volume, output_file)
        else:
            log.error(f"Source files not found : \
                                    {src_file_left} - {src_file_right}")

    def compute(self, number_subjects):
        """Loops over subjects and remove ventricle from volumes.
        """
        # Gets list of subjects
        log.debug(f"src_dir = {self.src_dir}")
        log.debug(f"reg exp = {self.expr}")

        if isdir(self.src_dir):
            filenames = glob.glob(join(f"{self.src_dir}", "L", "*.nii.gz")) + \
                glob.glob(join(f"{self.src_dir}", "R", "*.nii.gz"))
            
            log.debug(f"filenames = {filenames[:5]}")

            list_subjects = list(set(re.search(self.expr, basename(filename))[1]
                             for filename in filenames  
                             if is_it_a_subject(filename)))
            list_subjects = select_subjects(list_subjects, number_subjects)

            log.info(f"Expected number of subjects = {len(list_subjects)}")
            log.info(f"list_subjects[:5] = {list_subjects[:5]}")
            log.debug(f"list_subjects = {list_subjects}")
        else:
            raise NotADirectoryError(
                f"{self.src_dir} doesn't exist or is not a directory")

        # Performs computation on all subjects either serially or in parallel
        if self.parallel:
            log.info(
                "PARALLEL MODE: subjects are computed in parallel.")
            p_map(self.add_left_and_right_volumes_from_one_subject,
                  list_subjects,
                  num_cpus=define_njobs())
        else:
            log.info(
                "SERIAL MODE: subjects are scanned serially, "
                "without parallelism")
            for sub in list_subjects:
                self.add_left_and_right_volumes_from_one_subject(sub)

        # Checks if there is expected number of generated files
        compare_number_aims_files_with_expected(self.output_dir, list_subjects)


def add_left_and_right_volumes(src_dir=_SRC_DIR_DEFAULT,
                     src_filename=_SRC_FILENAME_DEFAULT,
                     output_filename=_OUTPUT_FILENAME_DEFAULT,
                     parallel=False,
                     number_subjects=_ALL_SUBJECTS):
    """ Add left and right volumes"""

    # Initialization
    adding = AddLeftandRightVolumes(
        src_dir=src_dir,
        src_filename=src_filename,
        output_filename=output_filename,
        parallel=parallel)
    adding.compute(number_subjects=number_subjects)


@exception_handler
def main(argv):
    """Reads argument line and add the two volumes
    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    add_left_and_right_volumes(
        src_dir=params["src_dir"],
        src_filename=params["src_filename"],
        output_filename=params["output_filename"],
        parallel=params['parallel'],
        number_subjects=params['nb_subjects'])


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
