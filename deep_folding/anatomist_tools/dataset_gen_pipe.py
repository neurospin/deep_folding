# -*- coding: utf-8 -*-
# /usr/bin/env python2.7 + brainvisa compliant env
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

"""Creating pickle file from T1 MRI datas

The aim of this script is to create dataset of cropped skeletons from MRIs
saved in a .pickle file.
Several steps are required: normalization, crop and .pickle generation

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python dataset_gen_pipe.py

  Alternatively, you can launch the script in the interactive terminal ipython:
  >>> %run dataset_gen_pipe.py

"""

from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import listdir
from os.path import join

import numpy as np

import six

from deep_folding.anatomist_tools.utils.logs import LogJson
from deep_folding.anatomist_tools.utils.load_bbox import compute_max_box
from deep_folding.anatomist_tools.utils.resample import resample
from deep_folding.anatomist_tools.load_data import fetch_data

_ALL_SUBJECTS = -1

_SIDE_DEFAULT = 'L'  # hemisphere 'L' or 'R'

_INTERP_DEFAULT = 'nearest'  # default interpolation for ApplyAimsTransform

_RESAMPLING_DEFAULT = None # if None, resampling method is AimsApplyTransform

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'S.T.s.ter.asc.ant._left'

# Input directories
# -----------------

# Input directory contaning the morphologist analysis of the HCP database
_SRC_DIR_DEFAULT = '/neurospin/hcp'

# Directory that contains the transformation file
# from native to MNI through SPM
# These files have been created with spm_skeleton
_TRANSFORM_DIR_DEFAULT = '/neurospin/dico/deep_folding_data/data/transform'

# Directory containing bounding box json files
_BBOX_DIR_DEFAULT = '/neurospin/dico/deep_folding_data/data/bbox'

# Output (target) directory
# -------------------------
_TGT_DIR_DEFAULT = '/neurospin/dico/deep_folding_data/test'


class DatasetCroppedSkeleton:
    """Generates cropped skeleton files and corresponding pickle file
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 transform_dir=_TRANSFORM_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 interp=_INTERP_DEFAULT,
                 resampling=_RESAMPLING_DEFAULT):
        """Inits with list of directories and list of sulci

        Args:
            src_dir: list of strings naming ful path source directories,
                    containing MRI images
            tgt_dir: name of target (output) directory with full path
            transform_dir: directory containing transformation files
                    (generated using transform.py)
            bbox_dir: directory containing bbox json files
                    (generated using bounding_box.py)
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left, or R for right hemisphere)
            interp: string giving interpolation for AimsApplyTransform
        """

        self.src_dir = src_dir

        # Transforms sulcus in a list of sulci
        self.list_sulci = ([list_sulci] if isinstance(list_sulci, str)
                           else list_sulci)

        self.tgt_dir = tgt_dir
        self.transform_dir = transform_dir
        self.bbox_dir = bbox_dir
        self.side = side
        self.interp = interp
        self.resampling = resampling
        if self.resampling:
            self.bbox_dir = '/neurospin/dico/deep_folding_data/test/bbox/resampling_bastien/'

        # Morphologist directory
        self.morphologist_dir = join(self.src_dir, "ANALYSIS/3T_morphologist")
        # default acquisition subdirectory
        self.acquisition_dir = "%(subject)s/t1mri/default_acquisition"
        # (input) name of normalized SPM file
        self.normalized_spm_file = "normalized_SPM_%(subject)s.nii"

        # Directory where to store cropped files
        self.cropped_dir = join(self.tgt_dir, self.side + 'crops')

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        # Files from morphologist pipeline
        self.normalized_spm_file = 'normalized_SPM_%(subject)s.nii'
        self.skeleton_file = 'default_analysis/segmentation/' \
                             '%(side)sskeleton_%(subject)s.nii.gz'

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        self.transform_file = 'natif_to_template_spm_%(subject)s.trm'
        self.cropped_file = '%(subject)s_normalized.nii.gz'

        # Initialization of bounding box coordinates
        self.bbmin = np.zeros(3)
        self.bbmax = np.zeros(3)

        # Creates json log class
        json_file = join(self.tgt_dir, self.side + 'dataset.json')
        self.json = LogJson(json_file)

    def crop_one_file(self, subject_id):
        """Crops one file

        Args:
            subject_id: string giving the subject ID
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        subject = {'subject': subject_id, 'side': self.side}

        # Names directory where subject analysis files are stored
        subject_dir = \
            join(self.morphologist_dir, self.acquisition_dir % subject)

        # Transformation file name
        file_transform = join(self.transform_dir, self.transform_file % subject)

        # Normalized SPM file name
        file_SPM = join(subject_dir, self.normalized_spm_file % subject)

        # Skeleton file name
        file_skeleton = join(subject_dir, self.skeleton_file % subject)
        # Creates output (cropped) file name
        file_cropped = join(self.cropped_dir, self.cropped_file % subject)

        # Normalization and resampling of skeleton images
        if self.resampling:
            resample(file_skeleton,
                     file_cropped,
                     output_vs=(2,2,2),
                     transformation=file_transform)

        else :
            cmd_normalize = 'AimsApplyTransform' + \
                            ' -i ' + file_skeleton + \
                            ' -o ' + file_cropped + \
                            ' -m ' + file_transform + \
                            ' -r ' + file_SPM + \
                            ' -t ' + self.interp
            os.system(cmd_normalize)

        # Take the coordinates of the bounding box
        bbmin = self.bbmin
        bbmax = self.bbmax
        xmin, ymin, zmin = str(bbmin[0]), str(bbmin[1]), str(bbmin[2])
        xmax, ymax, zmax = str(bbmax[0]), str(bbmax[1]), str(bbmax[2])

        # Crop of the images based on bounding box
        cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + \
                           ' -X ' + xmax + ' -Y ' + ymax + ' -Z ' + zmax
        cmd_crop = 'AimsSubVolume' + \
                   ' -i ' + file_cropped + \
                   ' -o ' + file_cropped + cmd_bounding_box
        os.system(cmd_crop)

    def crop_files(self, number_subjects=_ALL_SUBJECTS):
        """Crop nii files

        The programm loops over all subjects from the input (source) directory.

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

            # Creates target and cropped directory
            if not os.path.exists(self.tgt_dir):
                os.makedirs(self.tgt_dir)
            if not os.path.exists(self.cropped_dir):
                os.makedirs(self.cropped_dir)

            # Writes number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(list_subjects),
                           'src_dir': self.src_dir,
                           'transform_dir': self.transform_dir,
                           'bbox_dir': self.bbox_dir,
                           'side': self.side,
                           'interp': self.interp,
                           'list_sulci': self.list_sulci,
                           'bbmin': self.bbmin.tolist(),
                           'bbmax': self.bbmax.tolist(),
                           'tgt_dir': self.tgt_dir,
                           'cropped_dir': self.cropped_dir}
            self.json.update(dict_to_add=dict_to_add)

            for subject in list_subjects:
                self.crop_one_file(subject)

    def dataset_gen_pipe(self, number_subjects=_ALL_SUBJECTS):
        """Main API to create pickle files

        The programm loops over all subjects from the input (source) directory.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        self.json.write_general_info()

        # Generate cropped files
        if number_subjects:
            self.bbmin, self.bbmax = compute_max_box(sulci_list=self.list_sulci,
                                                     side=self.side,
                                                     talairach_box=False,
                                                     src_dir=self.bbox_dir)
        # Generate cropped files
        self.crop_files(number_subjects=number_subjects)
        # Creation of .pickle file for all subjects
        if number_subjects:
            fetch_data(cropped_dir=self.cropped_dir,
                       tgt_dir=self.tgt_dir,
                       side=self.side)


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='dataset_gen_pipe.py',
        description='Generates cropped and pickle files')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the MRI data lies. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the cropped and pickle files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-r", "--transform_dir", type=str, default=_TRANSFORM_DIR_DEFAULT,
        help='Transform directory where transformation files from native '
             'to Talairach files have been stored. '
             'Default is : ' + _TRANSFORM_DIR_DEFAULT)
    parser.add_argument(
        "-b", "--bbox_dir", type=str, default=_BBOX_DIR_DEFAULT,
        help='Bounding box directory where json files containing '
             'bounding box coordinates have been stored. '
             'Default is : ' + _BBOX_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT, nargs='+',
        help='Sulcus name around which we determine the bounding box. '
             'If there are several sulci, add all sulci '
             'one after the other. Example: -u sulcus_1 sulcus_2 '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        "-e", "--interp", type=str, default=_INTERP_DEFAULT,
        help="Same interpolation type as for AimsApplyTransform. "
             "Type of interpolation used for Volumes: "
             "n[earest], l[inear], q[uadratic], c[cubic], quartic, "
             "quintic, six[thorder], seven[thorder]. "
             "Modes may also be specified as order number: "
             "0=nearest, 1=linear...")
    parser.add_argument(
        "-p", "--resampling", type=str, default=None,
        help='Method of resampling to perform. '
             'Type of resampling: '
             's[ulcus] for Bastien method'
             'If None, AimsApplyTransform is used.'
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)
    params['src_dir'] = args.src_dir
    params['tgt_dir'] = args.tgt_dir
    params['bbox_dir'] = args.bbox_dir
    params['transform_dir'] = args.transform_dir
    params['list_sulci'] = args.sulcus  # a list of sulci
    params['side'] = args.side
    params['interp'] = args.interp
    params['resampling'] = args.resampling

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
            "number_subjects must be either the string \"all\" or an integer")
    params['nb_subjects'] = number_subjects

    return params


def dataset_gen_pipe(src_dir=_SRC_DIR_DEFAULT, tgt_dir=_TGT_DIR_DEFAULT,
                     transform_dir=_TRANSFORM_DIR_DEFAULT,
                     bbox_dir=_BBOX_DIR_DEFAULT, side=_SIDE_DEFAULT,
                     list_sulci=_SULCUS_DEFAULT, number_subjects=_ALL_SUBJECTS,
                     interp=_INTERP_DEFAULT, resampling=_RESAMPLING_DEFAULT):
    """Main program generating cropped files and corresponding pickle file
    """

    dataset = DatasetCroppedSkeleton(src_dir=src_dir, tgt_dir=tgt_dir,
                                     transform_dir=transform_dir,
                                     bbox_dir=bbox_dir, side=side,
                                     list_sulci=list_sulci, interp=interp,
                                     resampling=resampling)
    dataset.dataset_gen_pipe(number_subjects=number_subjects)


def main(argv):
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        params = parse_args(argv)
        # Actual API
        dataset_gen_pipe(src_dir=params['src_dir'],
                         tgt_dir=params['tgt_dir'],
                         transform_dir=params['transform_dir'],
                         bbox_dir=params['bbox_dir'],
                         side=params['side'],
                         list_sulci=params['list_sulci'],
                         interp=params['interp'],
                         number_subjects=params['nb_subjects'],
                         resampling=params['resampling'])
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
