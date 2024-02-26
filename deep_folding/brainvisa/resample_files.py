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

The aim of this script is to resample skeletons, foldlabels and distmaps.

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
from p_tqdm import p_map

import numpy as np

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.resample import resample
from deep_folding.brainvisa.utils.referentials import \
    generate_ref_volume_ICBM2009c
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    compare_number_aims_files_with_number_in_source, \
    get_not_processed_files, \
    save_list_to_csv
from pqdm.processes import pqdm
from deep_folding.config.logs import set_file_logger
from soma import aims

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _INPUT_TYPE_DEFAULT, _SKELETON_DIR_DEFAULT,\
    _TRANSFORM_DIR_DEFAULT, _RESAMPLED_SKELETON_DIR_DEFAULT,\
    _RESAMPLED_FOLDLABEL_DIR_DEFAULT, \
    _SIDE_DEFAULT, _VOXEL_SIZE_DEFAULT

_SKELETON_FILENAME = "skeleton_generated_"
_FOLDLABEL_FILENAME = "foldlabel_"
_DISTMAP_FILENAME = "distmap_generated_"
_RESAMPLED_SKELETON_FILENAME = "resampled_skeleton_"
_RESAMPLED_FOLDLABEL_FILENAME = "resampled_foldlabel_"
_RESAMPLED_DISTMAP_FILENAME = "resampled_distmap_"


# Defines logger
log = set_file_logger(__file__)

def mask_foldlabel(resampled,
                   skeleton_mask):
    """
    if do_skel=True,
    resampled foldlabel is masked using skeleton.
    """
    
    arr_resampled = resampled.np
    arr_skeleton = skeleton_mask.np
    arr = arr_resampled.copy()
    arr[arr_skeleton==0]=0
    vol = aims.Volume(arr)
    return vol


def resample_one_skeleton(input_image,
                          out_voxel_size,
                          transformation,
                          do_skel,
                          immortals):
    """Resamples one skeleton file

    Args
    ----
        input_image: either string or aims.Volume
            either path to skeleton or skeleton aims Volume
        out_voxel_size: tuple
            Output voxel size (default: None, no resampling)
        transformation: string or aims.Volume
            either path to transformation file or transformation itself
        do_skel: bool
            whether to apply skeletonization
        immortals: list
            if do_skel, then the list of immortal voxel values
            (NB: UNCLEAR)

    Returns:
        resampled: aims.Volume
            Transformed or resampled volume
    """

    # We give values with ascendent priority
    # The more important is the inversion in the priority
    # for the bottom value (30) and the simple surface value (60)
    # with respect to the natural order
    # We don't give background, which is the interior 0
    values = [11, 60, 30, 35, 10, 20, 40,
              50, 70, 80, 90, 100, 110, 120]

    # Normalization and resampling of skeleton images
    resampled = resample(input_image=input_image,
                         output_vs=out_voxel_size,
                         transformation=transformation,
                         values=values,
                         do_skel=do_skel,
                         immortals=immortals)
    return resampled


def resample_one_foldlabel(input_image,
                           out_voxel_size,
                           transformation,
                           do_skel=False,
                           immortals=None):
    """Resamples one foldlabel file

    Args
    ----
        input_image: either string or aims.Volume
            either path to foldlabel or foldlabel aims Volume
        out_voxel_size: tuple
            Output voxel size (default: None, no resampling)
        transformation: string or aims.Volume
            either path to transformation file or transformation itself

    Returns:
        resampled: aims.Volume
            Transformed or resampled volume
    """

    # Normalization and resampling of foldlabel images
    resampled = resample(input_image=input_image,
                         output_vs=out_voxel_size,
                         transformation=transformation)
    
    #if skeleton_mask is not None:
    #    resampled_masked = mask_foldlabel(resampled,
    #                                      skeleton_mask)
    #    return resampled_masked
    #else:
    #    return resampled
    return resampled


def resample_one_distmap(input_image,
                         resampled_dir,
                         out_voxel_size,
                         transformation):
    """Resamples one distmap file

    Args
    ----
        input_image: either string or aims.Volume
            either path to distmap or distmap aims Volume
        out_voxel_size: tuple
            Output voxel size (default: None, no resampling)
        transformation: string or aims.Volume
            either path to transformation file or transformation itself

    Returns:
            Save transformed or resampled volume
    """
    # temporary directory
    temp_dir = tempfile.mkdtemp()

    # transformation to native space of graph to ICBM template
    graph_to_icbm = aims.read(transformation)

    in_vol = aims.read(input_image)
    in_voxel_size = in_vol.header()['voxel_size']

    # definition of translation (half added voxels -> origin is in top left
    # corner)
    translation = (
        100 * in_voxel_size[0],
        100 * in_voxel_size[1],
        100 * in_voxel_size[2])
    distmap_to_padded_distmap = aims.AffineTransformation3d()
    distmap_to_padded_distmap.setTranslation(translation)

    # combination of translation with graph_to_icbm transformation
    padded_distmap_to_icbm = \
        graph_to_icbm * distmap_to_padded_distmap.inverse()

    transfo_file = f"{temp_dir}/padded_distmap_to_icbm.trm"
    aims.write(padded_distmap_to_icbm, transfo_file)

    ref_vol = generate_ref_volume_ICBM2009c(out_voxel_size)
    ref_file = f"{temp_dir}/ref_vol.nii.gz"
    aims.write(ref_vol, ref_file)

    # Normalization and resampling of skeleton images
    cmd_normalize = 'AimsApplyTransform' + \
        ' -i ' + input_image + \
        ' -o ' + resampled_dir + \
        ' -m ' + transfo_file + \
        ' -r ' + ref_file + \
        ' -t linear'
    print(cmd_normalize)
    os.system(cmd_normalize)


class FileResampler:
    """Resamples all files from source directories

    Parent class from which derive SkeletonResampler, FoldLabelResampler,...
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel,
                 do_skel, immortals
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing generated skeletons, labels or distmaps
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
        """
        self.side = side
        self.parallel = parallel

        # Names of files in function of dictionary: keys = 'subject' and 'side'
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
        
        self.do_skel = do_skel
        self.immortals = immortals

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str,
                             do_skel: bool,
                             immortals: list,
                             resampled_file=None):
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
        log.debug(f"resampled_file = {resampled_file}")

        # Performs the resampling
        if os.path.exists(src_file):
            resampled = self.resample_one_subject(
                src_file=src_file,
                out_voxel_size=self.out_voxel_size,
                transform_file=transform_file,
                resampled_file=resampled_file,
                do_skel=self.do_skel,
                immortals=self.immortals)
            # aims.write(resampled, resampled_file)
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

            log.debug(f"src_dir = {self.src_dir}")
            log.debug(f"reg exp = {self.expr}")

            if os.path.isdir(self.src_dir):
                src_files = glob.glob(f"{self.src_dir}/*.nii.gz")
                log.debug(f"list src files = {src_files}")
                log.debug(os.path.basename(src_files[0]))

                # Creates target directories
                create_folder(self.resampled_dir)

                # Generates list of subjects not treated yet
                not_processed_files = get_not_processed_files(
                    self.src_dir, self.resampled_dir, self.src_filename)

                list_all_subjects = \
                    [re.search(self.expr, os.path.basename(dI))[1]
                     for dI in not_processed_files]
            else:
                raise NotADirectoryError(
                    f"{self.src_dir} doesn't exist or is not a directory")

            if len(list_all_subjects):
                # Gives the possibility to list only the first number_subjects
                list_subjects = select_subjects_int(
                    list_all_subjects, number_subjects)
                log.info(f"Expected number of subjects = {len(list_subjects)}")
                log.info(f"list_subjects[:5] = {list_subjects[:5]}")
                log.debug(f"list_subjects = {list_subjects}")

                # Performs resampling for each file in a parallelized way
                if self.parallel:
                    log.info(
                        "PARALLEL MODE: subjects are in parallel")
                    p_map(
                        self.resample_one_subject_wrapper,
                        list_subjects,
                        num_cpus=define_njobs())
                else:
                    log.info(
                        "SERIAL MODE: subjects are scanned serially")
                    for sub in list_subjects:
                        self.resample_one_subject_wrapper(sub)
            else:
                list_subjects = []
                log.info(
                    "There is no subject or there is no subject to process "
                    "in the source directory")

            # Checks if there is expected number of generated files
            compare_number_aims_files_with_expected(self.resampled_dir,
                                                    list_subjects)

            # Checks if number of generated files == number of src files
            resampled_files, src_files = \
                compare_number_aims_files_with_number_in_source(
                    self.resampled_dir, self.src_dir)
            not_processed_files = get_not_processed_files(self.src_dir,
                                                          self.resampled_dir,
                                                          self.src_filename)
            save_list_to_csv(
                not_processed_files,
                f"{self.resampled_dir}/../not_processed_files.csv")


class SkeletonResampler(FileResampler):
    """Resamples all skeletons from source directories
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel, src_filename,
                 output_filename, do_skel, immortals
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing generated skeletons or labels
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
            src_filename : name of skeleton files
                          (format : "<SIDE><src_filename><SUBJECT>.nii.gz")
            output_filename : name of generated files
                          (format : "<SIDE><output_filename><SUBJECT>.nii.gz")
        """
        super(SkeletonResampler, self).__init__(
            src_dir=src_dir, resampled_dir=resampled_dir,
            transform_dir=transform_dir, side=side,
            out_voxel_size=out_voxel_size, parallel=parallel,
            do_skel=do_skel, immortals=immortals)
        
        #skeletonization parameters
        self.do_skel=do_skel
        self.immortals=immortals

        # Names of files in function of dictionary: keys = 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        # self.src_file = join(
        #    self.src_dir,
        #    '%(side)sskeleton_generated_%(subject)s.nii.gz')
        self.src_file = join(self.src_dir,
                             f'%(side)s' + src_filename + '%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.src_filename = src_filename
        self.resampled_file = join(
            self.resampled_dir,
            f'%(side)s' + output_filename + '%(subject)s.nii.gz')

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.' + src_filename + '(.*).nii.gz$'

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str,
                             do_skel: bool,
                             immortals: list,
                             resampled_file=None):
        """Resamples skeleton

        This static method is called by resample_one_subject_wrapper
        from parent class FileResampler"""
        resampled = resample_one_skeleton(input_image=src_file,
                                          out_voxel_size=out_voxel_size,
                                          transformation=transform_file,
                                          do_skel=do_skel,
                                          immortals=immortals)
        aims.write(resampled, resampled_file)


class FoldLabelResampler(FileResampler):
    """Resamples all files from source directories
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel, src_filename,
                 output_filename, do_skel, immortals
                 ):
        """Inits with list of directories

        Args:
            src_dir: folder containing generated skeletons or labels
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
            src_filename : name of fold label files
                          (format : "<SIDE><src_filename><SUBJECT>.nii.gz")
            output_filename : name of generated files
                          (format : "<SIDE><output_filename><SUBJECT>.nii.gz")
        """
        super(FoldLabelResampler, self).__init__(
            src_dir=src_dir, resampled_dir=resampled_dir,
            transform_dir=transform_dir, side=side,
            out_voxel_size=out_voxel_size, parallel=parallel,
            do_skel=do_skel, immortals=immortals)

        # Names of files in function of dictionary: keys = 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        self.src_file = join(
            self.src_dir,
            '%(side)s' + src_filename + '%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.src_filename = src_filename
        self.resampled_file = join(
            self.resampled_dir,
            f'%(side)s' + output_filename + '%(subject)s.nii.gz')

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.' + src_filename + '(.*).nii.gz$'

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str,
                             do_skel: bool,
                             immortals=None,
                             resampled_file=None):
        resampled = resample_one_foldlabel(input_image=src_file,
                                           out_voxel_size=out_voxel_size,
                                           transformation=transform_file,
                                           do_skel=do_skel)
        aims.write(resampled, resampled_file)


class DistMapResampler(FileResampler):
    """Resamples all files from source directories
    """

    def __init__(self, src_dir, resampled_dir, transform_dir,
                 side, out_voxel_size, parallel, src_filename,
                 output_filename, do_skel, immortals
                 ):
        """Inits with list of directories
        Args:
            src_dir: folder containing generated skeletons, labels or distmaps
            resampled_dir: name of target (output) directory,
            transform_dir: directory containing transform files to ICBM2009c
            side: either 'L' or 'R', hemisphere side
            out_voxel_size: float giving voxel size in mm
            parallel: does parallel computation if True
        """
        super(DistMapResampler, self).__init__(
            src_dir=src_dir, resampled_dir=resampled_dir,
            transform_dir=transform_dir, side=side,
            out_voxel_size=out_voxel_size, parallel=parallel,
            do_skel=do_skel, immortals=immortals)

        # Names of files in function of dictionary: keys = 'subject' and 'side'
        # Src directory contains either 'R' or 'L' a subdirectory
        self.src_file = join(
            self.src_dir,
            f'%(side)s' + src_filename + '%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.src_filename = src_filename
        self.resampled_file = join(
            self.resampled_dir,
            f'%(side)s' + output_filename + '%(subject)s.nii.gz')

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.' + src_filename + '(.*).nii.gz$'

    @staticmethod
    def resample_one_subject(src_file: str,
                             out_voxel_size: float,
                             transform_file: str,
                             resampled_file: str,
                             do_skel=None,
                             immortals=None):
        return resample_one_distmap(input_image=src_file,
                                    resampled_dir=resampled_file,
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
        help='Source directory where inputs files (skeletons, labels or '
             'distmaps) lie. '
             'Default is : ' + _SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-y", "--input_type", type=str, default=_INPUT_TYPE_DEFAULT,
        help='Input type: \'skeleton\', \'foldlabel\', \'distmap\' '
             'Default is : ' + _INPUT_TYPE_DEFAULT)
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=_RESAMPLED_SKELETON_DIR_DEFAULT,
        help='Target directory where to store the resampled files. '
        'Default is : ' +
        _RESAMPLED_SKELETON_DIR_DEFAULT)
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
        "-f", "--src_filename", type=str, default=_SKELETON_FILENAME,
        help='Filename of sources files. '
             'Format is : "<SIDE><src_filename><SUBJECT>.nii.gz" '
             'Default is : ' + _SKELETON_FILENAME)
    parser.add_argument(
        "-e",
        "--output_filename",
        type=str,
        default=_RESAMPLED_SKELETON_FILENAME,
        help='Filename of output files. '
        'Format is : "<SIDE><output_filename><SUBJECT>.nii.gz" '
        'Default is : ' +
        _RESAMPLED_SKELETON_FILENAME)
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
    params['src_filename'] = args.src_filename
    params['output_filename'] = args.output_filename
    return params


def resample_files(
        src_dir=_SKELETON_DIR_DEFAULT,
        input_type=_INPUT_TYPE_DEFAULT,
        resampled_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
        transform_dir=_TRANSFORM_DIR_DEFAULT,
        side=_SIDE_DEFAULT,
        out_voxel_size=_VOXEL_SIZE_DEFAULT,
        parallel=False,
        number_subjects=_ALL_SUBJECTS,
        src_filename=_SKELETON_FILENAME,
        output_filename=_RESAMPLED_SKELETON_FILENAME):

    if input_type == "skeleton":
        src_filename = (_SKELETON_FILENAME
                        if src_filename is None
                        else src_filename)
        output_filename = (_RESAMPLED_SKELETON_FILENAME
                           if output_filename is None
                           else output_filename)
        resampler = SkeletonResampler(
            src_dir=src_dir,
            resampled_dir=resampled_dir,
            transform_dir=transform_dir,
            side=side,
            out_voxel_size=out_voxel_size,
            parallel=parallel,
            src_filename=src_filename,
            output_filename=output_filename,
            do_skel=True,
            immortals=[30, 35, 100, 120])
    elif input_type == "foldlabel":
        src_filename = (_FOLDLABEL_FILENAME
                        if src_filename is None
                        else src_filename)
        output_filename = (_RESAMPLED_FOLDLABEL_FILENAME
                           if output_filename is None
                           else output_filename)
        resampler = FoldLabelResampler(
            src_dir=src_dir,
            resampled_dir=resampled_dir,
            transform_dir=transform_dir,
            side=side,
            out_voxel_size=out_voxel_size,
            parallel=parallel,
            src_filename=src_filename,
            output_filename=output_filename,
            do_skel=False,
            immortals=[])
    elif input_type == "distmap":
        resampler = DistMapResampler(
            src_dir=src_dir,
            resampled_dir=resampled_dir,
            transform_dir=transform_dir,
            side=side,
            out_voxel_size=out_voxel_size,
            parallel=parallel,
            src_filename=src_filename,
            output_filename=output_filename,
            do_skel=False,
            immortals=[])
    else:
        raise ValueError(
            "input_type: shall be either 'skeleton', 'foldlabel' or "
            "distmap")

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
        parallel=params['parallel'],
        src_filename=params['src_filename'],
        output_filename=params['output_filename']
    )


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
