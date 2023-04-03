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

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python write_distance_map.py


"""

import argparse
import glob
import os
import re
import sys

from pqdm.processes import pqdm
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.parallel import define_njobs

from deep_folding.config.logs import set_file_logger

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
        prog='write_distance_map.py',
        description='Generates distance maps from skeleton files')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source directory where the MRI data lies.')
    parser.add_argument(
        "-t", "--tgt_dir", type=str, required=True,
        help='Output directory where to put bucket files.')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')
    # Writes command line argument to target dir for logging
    setup_log(args,
              log_dir=f"{args.output_dir}/..",
              prog_name=os.path.basename(__file__),
              suffix='right' if args.side == 'R' else 'left')

    args = parser.parse_args(argv)

    return args


def skel_2_distMap(subject):
    """Reads volume, converts and writes back bucket.

    Args:
        vol_filename [str]: path to input volume file
        bucket_filename [str]: path to output bucket file
    """
    src_dir = "/neurospin/dico/data/deep_folding/datasets/hcp/skeleton/R"
    tgt_dir = "/neurospin/dico/data/deep_folding/datasets/hcp/distance_map/R"

    log.info(subject)
    skeleton_filename = f"{src_dir}/Rskeleton_generated_{subject}.nii.gz"
    distMap_filename = build_distMap_filename(subject, tgt_dir)

    cmd_distMap = 'VipDistanceMap' + \
        ' -i ' + skeleton_filename + \
        ' -o ' + distMap_filename + \
        ' -g f -d 0'
    log.debug(cmd_distMap)
    os.system(cmd_distMap)


def get_subject_name(filename):
    "Returns file basename without extension"
    subject = re.search('(\\d{6})', filename).group(1)
    return subject


def build_distMap_filename(subject, tgt_dir):
    """Returns bucket filename"""
    return f"{tgt_dir}/distance_map_{subject}.nii.gz"


def loop_over_directory(src_dir, tgt_dir):
    """Loops conversion over input directory
    """
    # Gets and creates all filenames
    filenames = glob.glob(f"{src_dir}/*.nii.gz")
    subjects = [get_subject_name(filename) for filename in filenames]
    log.info(f"subjects[:5] = {subjects[:5]}")
    log.debug(subjects)
    # distMap_filenames = [build_distMap_filename(subject, tgt_dir)
    # for subject in subjects]

    # for sub in tqdm(subjects):
    #    skel_2_distMap(sub)

    pqdm(subjects, skel_2_distMap, n_jobs=define_njobs())


@exception_handler
def main(argv):
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    args = parse_args(argv)
    loop_over_directory(args.src_dir, args.tgt_dir)


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])

    # Example of use:
    # python3 convert_volume_to_bucket.py \
    # -s /neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/simple_combined/Rcrops \
    # -t /neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/simple_combined/Rbuckets
