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

"""Write distance maps from skeleton files

   Note: generation of distance maps should be performed for each resolution

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python generate_distmap.py


"""

import argparse
import glob
import re
import sys
import os
from os.path import abspath
from os.path import basename
from os.path import normpath
from os import pardir

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.distmap import \
    generate_distmap_from_skeleton_file,\
    generate_distmap_from_resampled_skeleton
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_subjects_distmap
from pqdm.processes import pqdm
from p_tqdm import p_map
from deep_folding.config.logs import set_file_logger

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SKELETON_DIR_DEFAULT,\
    _DISTMAPS_DIR_DEFAULT, _SIDE_DEFAULT

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
        description='Generates distance maps files from skeleton files')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SKELETON_DIR_DEFAULT,
        help='Source directory where the skeleton data lies. '
             'Default is : ' + _SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_DISTMAPS_DIR_DEFAULT,
        help='Output directory where to put distmap files.'
        'Default is : ' + _DISTMAPS_DIR_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-r", "--resampled_skel", default=False,
        help='if set (-a), generates distmap from already resampled skeleton')
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
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix='right' if args.side == 'R' else 'left')

    params = {}

    params['src_dir'] = abspath(args.src_dir)
    params['output_dir'] = abspath(args.output_dir)
    params['side'] = args.side
    params['parallel'] = args.parallel
    params['resampled_skel'] = args.resampled_skel
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


class SkelConvert2DistMap:
    """Class to convert all skeletons from a folder into distance maps

    It contains all information to scan a dataset for skeletons
    and writes distance maps into target directory
    """

    def __init__(self, src_dir, distmaps_dir,
                 side, parallel, resampled_skel=False):
        self.src_dir = src_dir
        self.distmap_dir = distmaps_dir
        self.side = side
        self.parallel = parallel
        self.distmap_dir = f"{self.distmap_dir}/{self.side}"
        self.resampled_skel = resampled_skel

        create_folder(abspath(self.distmap_dir))

    def generate_one_distmap(self, subject: str):
        """Generates and writes distmap for one subject.
        """
        distmap_file = f"{self.distmap_dir}/" +\
            f"{self.side}distmap_generated_{subject}.nii.gz"

        if self.resampled_skel:
            skeleton_file = glob.glob(f"{self.src_dir}/" +
                                      f"*{subject}*.nii.gz")[0]
            generate_distmap_from_resampled_skeleton(skeleton_file,
                                                     distmap_file)
        else:
            skeleton_file = glob.glob(f"{self.src_dir}/{self.side}/" +
                                      f"*{subject}*.nii.gz")[0]
            generate_distmap_from_skeleton_file(skeleton_file,
                                                distmap_file)

    def compute(self, number_subjects):
        """Loops over subjects and converts graphs into distmaps.
        """
        # Gets list fo subjects
        filenames = glob.glob(f"{self.src_dir}/{self.side}/*.nii.gz")

        list_subjects = [
            re.search(
                f"({self.side}skeleton_generated_)(.*)(\\.nii\\.gz)",
                filename).group(2) for filename in filenames]
        list_subjects = select_subjects_int(list_subjects, number_subjects)
        log.info(f"list_subjects[:5] before = {list_subjects[:5]}")
        list_subjects = \
            get_not_processed_subjects_distmap(list_subjects, self.distmap_dir)
        log.info(f"Expected number of subjects = {len(list_subjects)}")
        log.info(f"list_subjects[:5] = {list_subjects[:5]}")
        log.debug(f"list_subjects = {list_subjects}")

        # Performs computation on all subjects either serially or in parallel
        if self.parallel:
            log.info(
                "PARALLEL MODE: subjects are computed in parallel.")

            p_map(self.generate_one_distmap,
                  list_subjects,
                  num_cpus=20)
        else:
            log.info(
                "SERIAL MODE: subjects are scanned serially, "
                "without parallelism")
            for sub in list_subjects:
                self.generate_one_distmap(sub)

        # Checks if there is expected number of generated files
        compare_number_aims_files_with_expected(self.distmap_dir,
                                                list_subjects)


def generate_distmaps(
        src_dir=_SKELETON_DIR_DEFAULT,
        distmaps_dir=_DISTMAPS_DIR_DEFAULT,
        side=_SIDE_DEFAULT,
        parallel=False,
        resampled_skel=True,
        number_subjects=_ALL_SUBJECTS):
    """Generates distmaps from skeletons"""

    # Initialization
    conversion = SkelConvert2DistMap(
        src_dir=src_dir,
        distmaps_dir=distmaps_dir,
        side=side,
        parallel=parallel,
        resampled_skel=resampled_skel
    )
    # Actual generation of skeletons from graphs
    conversion.compute(number_subjects=number_subjects)


@exception_handler
def main(argv):
    """Reads argument line and generates distmaps from skeletons

    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    generate_distmaps(
        src_dir=params['src_dir'],
        distmaps_dir=params['output_dir'],
        side=params['side'],
        parallel=params['parallel'],
        resampled_skel=params['resampled_skel'],
        number_subjects=params['nb_subjects'])


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
