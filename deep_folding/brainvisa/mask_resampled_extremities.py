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

"""Masks extremities

The aim of this script is to mask extremities using re-skeletized skeletons.

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python mask_resampled_extremities.py

  Alternatively, you can launch the script in the interactive terminal ipython:
  >>> %run mask_resampled_extremities.py

"""

import argparse
import glob
import os
import re
import sys
from os.path import join
from os.path import basename
from p_tqdm import p_map

import numpy as np
import warnings

from skimage.morphology import ball, binary_dilation

from soma import aims

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.parallel import define_njobs

from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    compare_number_aims_files_with_number_in_source, \
    get_not_processed_files, \
    save_list_to_csv
from deep_folding.config.logs import set_file_logger

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, \
    _RESAMPLED_SKELETON_DIR_DEFAULT, \
    _RESAMPLED_EXTREMITIES_DIR_DEFAULT, \
    _SIDE_DEFAULT


# Defines logger
log = set_file_logger(__file__)

_VX_TOLERANCE = 2
_DILATION_MAGNITUDE = 2


def nearest_nonzero_idx(a, x, y, z):
    tmp = a[x, y, z]
    a[x, y, z] = 0
    d, e, f = np.nonzero(a)
    a[x, y, z] = tmp
    min_idx = ((d - x)**2 + (e - y)**2 + (f - z)**2).argmin()
    return (d[min_idx], e[min_idx], f[min_idx])


class ExtremitiesMasker:
    """Maskes extremities using reskeletized skeletons as masks

    """

    def __init__(self, src_dir, skeleton_dir, masked_dir,
                 side, parallel
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing resampled extremities
            skeleton_dir: folder containing resampled + reskeletized skeletons
            masked_dir: name of target (output) directory,
            side: either 'L' or 'R', hemisphere side
            parallel: does parallel computation if True
        """
        self.side = side
        self.parallel = parallel

        # Names of files in function of dictionary: keys = 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        self.src_dir = join(src_dir, self.side) + "_before_masking"

        self.skeleton_dir = join(skeleton_dir, self.side)

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.masked_dir = join(masked_dir, self.side)

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.resampled_extremities_(.*).nii.gz$'

        self.src_filename = f"{self.side}resampled_extremities_"
        self.output_filename = f"{self.side}resampled_extremities_"

    def mask_one_file(self, subject: str):
        skel = aims.read(
            os.path.join(
                self.skeleton_dir,
                f'{self.side}resampled_skeleton_{subject}.nii.gz'))
        old_extremities = aims.read(
            os.path.join(
                self.src_dir,
                f'{self.side}resampled_extremities_{subject}.nii.gz'))
        skel_np = skel.np
        old_extremities_np = old_extremities.np

        extremities = old_extremities_np.copy()
        # first mask skeleton using extremities because sometimes
        # 1vx is added during skeletonization...
        extremities[skel_np == 0] = 0

        # # extremities are computed without the top (value 35 in skeleton)
        # # we here add the top, by looking for the value 35 in skeleton
        # # at distance 1 of the extremities
        # log.info("Extremities voxels without tops: "
        #          f"{np.sum(old_extremities_np2)}")
        # extremities_dilated = binary_dilation(
        #                         old_extremities_np2[:, :, :, 0],
        #                         ball(_DILATION_MAGNITUDE))
        # extremities_dilated = np.expand_dims(extremities_dilated, axis=-1)
        # tops_of_extremities = np.logical_and(extremities_dilated,
        #                                      (skel_np == 35))
        # extremities = np.logical_or(old_extremities_np2,
        #                             tops_of_extremities)
        # extremities = extremities.astype(np.int16)
        # print("Voxels of extremities after adding tops : "
        #       f"{np.sum(extremities)}")

        # Makes som checks
        f = extremities != 0
        s = skel_np.copy()
        skel_np[extremities == 0] = 0
        s = skel_np != 0
        diff_fs = np.sum(f != s)
        assert (diff_fs <= _VX_TOLERANCE), (
            f"subject {subject} has incompatible extremities and skeleton."
            f"{np.sum(s)} vx in skeleton, {np.sum(f)} vx in extremities"
        )
        if diff_fs != 0:
            warnings.warn(
                f"subject {subject} has incompatible extremities/skeleton. "
                f"{np.sum(s)} vx in skeleton, {np.sum(f)} vx in extremities")
            idxs = np.where(f != s)
            print(idxs)
            for i in range(diff_fs):
                x, y, z = idxs[0][i], idxs[1][i], idxs[2][i]
                d, e, f = nearest_nonzero_idx(extremities[:, :, :, 0], x, y, z)
                extremities[x, y, z, 0] = extremities[d, e, f, 0]
                print(f"extremities has a 0 at index {x,y,z}, "
                      f"nearest nonzero at index {d,e,f}, "
                      f"value {extremities[d,e,f,0]}")
        f = extremities != 0
        assert np.sum(f != s) == 0, \
            f"subject {subject} has incompatible extremities and skeleton " \
            "AFTER CORRECTION. " \
            f"{np.sum(s)} vx in skeleton, {np.sum(f)} vx in extremities"

        # Writes volume to file
        vol = aims.Volume(extremities)
        vol.header()['voxel_size'] = old_extremities.header()['voxel_size']
        aims.write(
            vol,
            os.path.join(self.masked_dir,
                         f'{self.side}resampled_extremities_{subject}.nii.gz'))

    def compute(self, nb_subjects=_ALL_SUBJECTS):
        """Loops over nii files

        The programm loops over all subjects from the input (source) directory.

        Args:
            nb_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if nb_subjects:

            log.debug(f"src_dir = {self.src_dir}")
            log.debug(f"reg exp = {self.expr}")

            if os.path.isdir(self.src_dir):
                src_files = glob.glob(f"{self.src_dir}/*.nii.gz")
                log.debug(f"list src files = {src_files}")
                log.debug(os.path.basename(src_files[0]))

                # Creates target directories
                create_folder(self.masked_dir)

                # Generates list of subjects not treated yet
                not_processed_files = get_not_processed_files(
                    self.src_dir, self.masked_dir, self.src_filename)

                list_not_processed_subjects = [
                    re.search(self.expr, basename(dI))[1]
                    for dI in not_processed_files]
                list_all_subjects = [
                    re.search(self.expr, basename(dI))[1]
                    for dI in src_files]
            else:
                raise NotADirectoryError(
                    f"{self.src_dir} doesn't exist or is not a directory")

            if len(list_all_subjects):
                log.info(f"First subject to process: {list_all_subjects[0]}")
                log.info(
                    "Number of requested subjects: "
                    f"{nb_subjects}, {type(nb_subjects)}")
                # Gives the possibility to list only the first nb_subjects
                list_subjects = select_subjects_int(
                                    list_all_subjects,
                                    list_not_processed_subjects,
                                    nb_subjects)
                log.info(f"Expected number of subjects = {len(list_subjects)}")
                log.info(f"list_subjects[:5] = {list_subjects[:5]}")
                log.debug(f"list_subjects = {list_subjects}")

                # Performs resampling for each file in a parallelized way
                if self.parallel:
                    log.info(
                        "PARALLEL MODE: subjects are in parallel")
                    p_map(
                        self.mask_one_file,
                        list_subjects,
                        num_cpus=define_njobs())
                else:
                    log.info(
                        "SERIAL MODE: subjects are scanned serially")
                    for sub in list_subjects:
                        self.mask_one_file(sub)
            else:
                list_subjects = []
                log.info(
                    "There is no subject or there is no subject to process "
                    "in the source directory")

            # Checks if there is expected number of generated files
            compare_number_aims_files_with_expected(self.masked_dir,
                                                    list_subjects)

            # Checks if number of generated files == number of src files
            masked_files, src_files = \
                compare_number_aims_files_with_number_in_source(
                    self.masked_dir, self.src_dir)
            not_processed_files = get_not_processed_files(self.src_dir,
                                                          self.masked_dir,
                                                          self.src_filename)
            save_list_to_csv(
                not_processed_files,
                f"{self.masked_dir}/../not_processed_files.csv")


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description='Masks resampled extremities files')
    parser.add_argument(
        "-s", "--src_dir", type=str,
        default=_RESAMPLED_EXTREMITIES_DIR_DEFAULT,
        help='Source directory where input resampled extremities files lie. '
             'Default is : ' + _RESAMPLED_EXTREMITIES_DIR_DEFAULT)
    parser.add_argument(
        "-k", "--skeleton_dir", type=str,
        default=_RESAMPLED_SKELETON_DIR_DEFAULT,
        help='Source directory where input resampled skeleton files lie. '
             'Default is : ' + _RESAMPLED_SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=_RESAMPLED_EXTREMITIES_DIR_DEFAULT,
        help='Target directory where to store the masked files. '
        'Default is : ' +
        _RESAMPLED_EXTREMITIES_DIR_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
        'If no option is provided then logging.INFO is selected. '
        'If one option -v (or -vv) or more is provided '
        'then logging.DEBUG is selected.')

    params = {}

    args = parser.parse_args(argv)

    setup_log(args,
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix='right' if args.side == 'R' else 'left')

    params = vars(args)

    params['masked_dir'] = args.output_dir
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


def mask_extremities_files(
        src_dir=_RESAMPLED_EXTREMITIES_DIR_DEFAULT,
        skeleton_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
        masked_dir=_RESAMPLED_EXTREMITIES_DIR_DEFAULT,
        side=_SIDE_DEFAULT,
        parallel=False,
        nb_subjects=_ALL_SUBJECTS):

    masker = ExtremitiesMasker(
        src_dir=src_dir,
        skeleton_dir=skeleton_dir,
        masked_dir=masked_dir,
        side=side,
        parallel=parallel
    )

    masker.compute(nb_subjects=nb_subjects)


@exception_handler
def main(argv):
    """Reads argument line and resamples files

    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    mask_extremities_files(
        src_dir=params['src_dir'],
        skeleton_dir=params['skeleton_dir'],
        masked_dir=params['masked_dir'],
        side=params['side'],
        nb_subjects=params['nb_subjects'],
        parallel=params['parallel']
    )


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
