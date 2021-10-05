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

from soma import aims

from pqdm.processes import pqdm
from joblib import cpu_count

from deep_folding.anatomist_tools.utils.logs import LogJson
from deep_folding.anatomist_tools.utils.load_bbox import compute_max_box
from deep_folding.anatomist_tools.utils.resample import resample
from deep_folding.anatomist_tools.utils.sulcus_side import complete_sulci_name
from deep_folding.anatomist_tools.load_data import fetch_data

from tqdm import tqdm

_ALL_SUBJECTS = -1

_SIDE_DEFAULT = 'L'  # hemisphere 'L' or 'R'

_INTERP_DEFAULT = 'nearest'  # default interpolation for ApplyAimsTransform

_RESAMPLING_DEFAULT = None # if None, resampling method is AimsApplyTransform

_OUT_VOXEL_SIZE = (1, 1, 1) # default output voxel size for Bastien's resampling

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'S.T.s.ter.asc.ant.'

# Input directories
# -----------------

# Input directory contaning the morphologist analysis of the HCP database
_SRC_DIR_DEFAULT = '/neurospin/hcp'

# Directory where subjects to be processed are stored.
# Default is for HCP dataset
_MORPHOLOGIST_DIR_DEFAULT = 'ANALYSIS/3T_morphologist'

# Directory that contains the transformation file
# from native to MNI through SPM
# These files have been created with spm_skeleton
_TRANSFORM_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/data/transform'

# Directory containing bounding box json files
# default corresponds to bounding boxes computed for voxels of 1mm
_BBOX_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/data/bbox'

# Directory containing bounding box json files
# default corresponds to bounding boxes computed for voxinput 
# -------------------------
_TGT_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/test'

def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus-2, 1)

class DatasetCroppedSkeleton:
    """Generates cropped skeleton files and corresponding pickle file
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 transform_dir=_TRANSFORM_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 morphologist_dir=_MORPHOLOGIST_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 interp=_INTERP_DEFAULT,
                 resampling=_RESAMPLING_DEFAULT,
                 out_voxel_size=_OUT_VOXEL_SIZE):
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
        self.side = side
        # Transforms sulcus in a list of sulci
        self.list_sulci = ([list_sulci] if isinstance(list_sulci, str)
                           else list_sulci)
        self.list_sulci = complete_sulci_name(self.list_sulci, self.side)

        self.tgt_dir = tgt_dir
        self.transform_dir = transform_dir
        self.bbox_dir = bbox_dir
        self.morphologist_dir = morphologist_dir
        self.interp = interp
        self.resampling = resampling
        self.out_voxel_size = out_voxel_size

        # Morphologist directory
        self.morphologist_dir = join(self.src_dir, self.morphologist_dir)
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
        if os.path.exists(file_skeleton) and os.path.exists(file_transform):
            # Creates output (cropped) file name
            file_cropped = join(self.cropped_dir, self.cropped_file % subject)

            # Normalization and resampling of skeleton images
            if self.resampling:
                resampled = resample(input_image=file_skeleton,
                                     output_vs=self.out_voxel_size,
                                     transformation=file_transform,
                                     verbose=False)
                aims.write(resampled, file_cropped)
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
            
            # Sts output from AimsSubVolume is recorded in var_output
            # Put following command to get the output
            # os.popen(cmd_crop).read()
            var_output = os.popen(cmd_crop).read()

    def crop_files(self, number_subjects=_ALL_SUBJECTS):
        """Crop nii files

        The programm loops over all subjects from the input (source) directory.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if number_subjects:

            # subjects are detected as the directory names under src_dir
            list_all_subjects = [dI for dI in os.listdir(self.morphologist_dir)\
             if os.path.isdir(os.path.join(self.morphologist_dir,dI))]

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
                           'cropped_dir': self.cropped_dir,
                           'resampling_type': 'AimsApplyTransform' if self.resampling is None else 'Bastien',
                           'out_voxel_size': self.out_voxel_size
                           }
            
            self.json.update(dict_to_add=dict_to_add)

            print(list_subjects)

            pqdm(list_subjects, self.crop_one_file, n_jobs=define_njobs())
            #Parallel(n_jobs = define_njobs())(delayed(self.crop_one_file)(subject)
            #                                  for subject in list_subjects)

    def dataset_gen_pipe(self, number_subjects=_ALL_SUBJECTS):
        """Main API to create pickle files

        The programm loops over all subjects from the input (source) directory.
            # Writes number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(list_subjects),joblib import Parallel, delayed
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
        "-m", "--morphologist_dir", type=str, default=_MORPHOLOGIST_DIR_DEFAULT,
        help='Directory where subjects to be processed are stored')
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
    parser.add_argument(
        "-v", "--out_voxel_size", type=int, nargs='+', default=_OUT_VOXEL_SIZE,
        help='Voxel size of output images'
             'Default is : 1 1 1')

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
    params['out_voxel_size'] = tuple(args.out_voxel_size)
    params['morphologist_dir'] = args.morphologist_dir

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
                     bbox_dir=_BBOX_DIR_DEFAULT,
                     morphologist_dir=_MORPHOLOGIST_DIR_DEFAULT,
                     side=_SIDE_DEFAULT, list_sulci=_SULCUS_DEFAULT,
                     number_subjects=_ALL_SUBJECTS, interp=_INTERP_DEFAULT,
                     resampling=_RESAMPLING_DEFAULT,
                     out_voxel_size=_OUT_VOXEL_SIZE):
    """Main program generating cropped files and corresponding pickle file
    """

    dataset = DatasetCroppedSkeleton(src_dir=src_dir, tgt_dir=tgt_dir,
                                     transform_dir=transform_dir,
                                     bbox_dir=bbox_dir,
                                     morphologist_dir=morphologist_dir,
                                     side=side, list_sulci=list_sulci,
                                     interp=interp, resampling=resampling,
                                     out_voxel_size=out_voxel_size)
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
                         morphologist_dir=params['morphologist_dir'],
                         side=params['side'],
                         list_sulci=params['list_sulci'],
                         interp=params['interp'],
                         number_subjects=params['nb_subjects'],
                         resampling=params['resampling'],
                         out_voxel_size=params['out_voxel_size'])
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
