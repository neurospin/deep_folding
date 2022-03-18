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

"""Resample skeletons

The aim of this script is to resample skeletons.

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python resample_files.py

  Alternatively, you can launch the script in the interactive terminal ipython:
  >>> %run resample_files.py

"""

import argparse
from asyncio.subprocess import DEVNULL
from email.mime import base
import glob
import os
import re
import sys
import tempfile
from os.path import join
from os.path import basename

import numpy as np

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.resample import resample
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected
from pqdm.processes import pqdm
from deep_folding.config.logs import set_file_logger
from soma import aims

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _INPUT_TYPE_DEFAULT, _SKELETON_DIR_DEFAULT,\
    _TRANSFORM_DIR_DEFAULT, _RESAMPLED_SKELETON_DIR_DEFAULT,\
    _SIDE_DEFAULT, _VOXEL_SIZE_DEFAULT

# Defines logger
log = set_file_logger(__file__)

# temporary directory
temp_dir = tempfile.mkdtemp()


def resample_one_skeleton(input_image,
                          out_voxel_size,
                          transformation):
    """Resamples one skeleton file

    Args
    ----
        input_image: either string or aims.Volume
            either path to skeleton or skeleton aims Volume
        out_voxel_size: tuple
            Output voxel size (default: None, no resampling)
        transformation: string or aims.Volume
            either path to transformation file or transformation itself

    Returns:
        resampled: aims.Volume
            Transformed or resampled volume
    """

    # We give values with ascendent priority
    # The more important is the inversion in the priority
    # for the bottom value (30) and the simple surface value (60)
    # with respect to the natural order
    # We don't give background, which is the interior 0
    values = np.array([11, 60, 30, 10, 20, 40, 50, 70, 80, 90])

    # Normalization and resampling of skeleton images
    resampled = resample(input_image=input_image,
                         output_vs=out_voxel_size,
                         transformation=transformation,
                         values=values)
    return resampled


def resample_one_foldlabel(input_image,
                           out_voxel_size,
                           transformation):
    """Resamples one foldlabel file

    Args
    ----
        input_image: either string or aims.Volume
            either path to skeleton or skeleton aims Volume
        out_voxel_size: tuple
            Output voxel size (default: None, no resampling)
        transformation: string or aims.Volume
            either path to transformation file or transformation itself

    Returns:
        resampled: aims.Volume
            Transformed or resampled volume
    """

    # We have given values with ascendent priority
    # Higher values have more priority than smaller values

    # Normalization and resampling of skeleton images
    resampled = resample(input_image=input_image,
                         output_vs=out_voxel_size,
                         transformation=transformation)
    return resampled


class FileResampler:
    """Resamples all files from source directories

    Parent class from which derive SkeletonResampler, FoldLabelResampler,...
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing generated skeletons or labels
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
        """
        self.side = side
        self.parallel = parallel

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        self.src_dir = join(src_dir, self.side)

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.resampled_dir = join(resampled_dir, self.side)

        # transform_dir contains side 'R' or 'L'
        self.transform_dir = join(transform_dir, self.side)
        self.transform_file = join(
            self.transform_dir,
            '%(side)stransform_to_ICBM2009c_%(subject)s.trm')

        self.out_voxel_size = (out_voxel_size,
                               out_voxel_size,
                               out_voxel_size)

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str):
        """Resamples skeleton

        This static method is called by resample_one_subject_wrapper
        from parent class FileResampler"""
        raise RuntimeError(
            "Method from parent class FileResampler. "
            "Shall be implemented and called only from child class.")

    def resample_one_subject_wrapper(self, subject_id):
        """Resamples one file

        Args:
            subject_id: string giving the subject ID
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        subject = {'subject': subject_id, 'side': self.side}

        # Creates transformation MNI template
        transform_file = self.transform_file % subject

        # Input raw file name
        src_file = self.src_file % subject

        # Output resampled file name
        resampled_file = self.resampled_file % subject

        # Performs the resampling
        if os.path.exists(src_file):
            resampled = self.resample_one_subject(
                src_file=src_file,
                out_voxel_size=self.out_voxel_size,
                transform_file=transform_file)
            aims.write(resampled, resampled_file)
        else:
            raise FileNotFoundError(f"{src_file} not found")

    def compute(self, number_subjects=_ALL_SUBJECTS):
        """Loops over nii files

        The programm loops over all subjects from the input (source) directory.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if number_subjects:

            if os.path.isdir(self.src_dir):
                list_all_subjects = \
                    [re.search(self.expr, os.path.basename(dI))[1]
                     for dI in glob.glob(f"{self.src_dir}/*.nii.gz")]
            else:
                raise NotADirectoryError(
                    f"{self.src_dir} doesn't exist or is not a directory")

            # Gives the possibility to list only the first number_subjects
            list_subjects = select_subjects_int(
                list_all_subjects, number_subjects)
            log.info(f"Expected number of subjects = {len(list_subjects)}")
            log.info(f"list_subjects[:5] = {list_subjects[:5]}")
            log.debug(f"list_subjects = {list_subjects}")

            # Creates target directories
            create_folder(self.resampled_dir)

            # Performs resampling for each file in a parallelized way
            if self.parallel:
                log.info(
                    "PARALLEL MODE: subjects are in parallel")
                pqdm(
                    list_subjects,
                    self.resample_one_subject_wrapper,
                    n_jobs=define_njobs())
            else:
                log.info(
                    "SERIAL MODE: subjects are scanned serially")
                for sub in list_subjects:
                    self.resample_one_subject_wrapper(sub)

            # Checks if there is expected number of generated files
            compare_number_aims_files_with_expected(self.resampled_dir,
                                                    list_subjects)


class SkeletonResampler(FileResampler):
    """Resamples all skeletons from source directories
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing generated skeletons or labels
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
        """
        super(SkeletonResampler, self).__init__(
            src_dir=src_dir, resampled_dir=resampled_dir,
            transform_dir=transform_dir, side=side,
            out_voxel_size=out_voxel_size, parallel=parallel)

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        self.src_file = join(
            self.src_dir,
            '%(side)sskeleton_generated_%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.resampled_file = join(
            self.resampled_dir,
            f'%(side)sresampled_skeleton_%(subject)s.nii.gz')

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.skeleton_generated_([0-9a-zA-Z]*).nii.gz$'

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str):
        """Resamples skeleton

        This static method is called by resample_one_subject_wrapper
        from parent class FileResampler"""
        return resample_one_skeleton(input_image=src_file,
                                     out_voxel_size=out_voxel_size,
                                     transformation=transform_file)


class FoldLabelResampler(FileResampler):
    """Resamples all files from source directories
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing generated skeletons or labels
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
        """
        super(FoldLabelResampler, self).__init__(
            src_dir=src_dir, resampled_dir=resampled_dir,
            transform_dir=transform_dir, side=side,
            out_voxel_size=out_voxel_size, parallel=parallel)

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        self.src_file = join(
            self.src_dir,
            '%(side)sfoldlabel_%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.resampled_file = join(
            self.resampled_dir,
            f'%(side)sresampled_foldlabel_%(subject)s.nii.gz')

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.foldlabel_([0-9a-zA-Z]*).nii.gz$'

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str):
        return resample_one_foldlabel(input_image=src_file,
                                      out_voxel_size=out_voxel_size,
                                      transformation=transform_file)


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
        description='Generates resampled files (skeletons, foldlabels,...)')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SKELETON_DIR_DEFAULT,
        help='Source directory where inputs files (skeletons or labels) lie. '
             'Default is : ' + _SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-y", "--input_type", type=str, default=_INPUT_TYPE_DEFAULT,
        help='Input type: \'skeleton\', \'foldlabel\' '
             'Default is : ' + _INPUT_TYPE_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_RESAMPLED_SKELETON_DIR_DEFAULT,
        help='Target directory where to store the resampled files. '
             'Default is : ' + _RESAMPLED_SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--transform_dir", type=str, default=_TRANSFORM_DIR_DEFAULT,
        help='Transform directory containing transform files to ICBM2009c. '
             'Default is : ' + _TRANSFORM_DIR_DEFAULT)
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
        "-x", "--out_voxel_size", type=float, default=_VOXEL_SIZE_DEFAULT,
        help='Voxel size of bounding box. '
             'Default is : None')
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

    params['src_dir'] = args.src_dir
    params['input_type'] = args.input_type
    params['resampled_dir'] = args.output_dir
    params['transform_dir'] = args.transform_dir
    params['side'] = args.side
    params['out_voxel_size'] = args.out_voxel_size
    params['parallel'] = args.parallel
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


def resample_files(
        src_dir=_SKELETON_DIR_DEFAULT,
        input_type=_INPUT_TYPE_DEFAULT,
        resampled_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
        transform_dir=_TRANSFORM_DIR_DEFAULT,
        side=_SIDE_DEFAULT,
        out_voxel_size=_VOXEL_SIZE_DEFAULT,
        parallel=False,
        number_subjects=_ALL_SUBJECTS):

    if input_type == "skeleton":
        resampler = SkeletonResampler(
            src_dir=src_dir,
            resampled_dir=resampled_dir,
            transform_dir=transform_dir,
            side=side,
            out_voxel_size=out_voxel_size,
            parallel=parallel)
    elif input_type == "foldlabel":
        resampler = FoldLabelResampler(
            src_dir=src_dir,
            resampled_dir=resampled_dir,
            transform_dir=transform_dir,
            side=side,
            out_voxel_size=out_voxel_size,
            parallel=parallel)
    else:
        raise ValueError(
            "input_type: shall be either 'skeleton' or 'foldlabel'")

    resampler.compute(number_subjects=number_subjects)


@exception_handler
def main(argv):
    """Reads argument line and resamples files

    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    resample_files(
        src_dir=params['src_dir'],
        input_type=params['input_type'],
        resampled_dir=params['resampled_dir'],
        transform_dir=params['transform_dir'],
        side=params['side'],
        number_subjects=params['nb_subjects'],
        out_voxel_size=params['out_voxel_size'],
        parallel=params['parallel'])


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
