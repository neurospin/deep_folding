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

import argparse
import glob
import os
import re
import sys
import tempfile
from os import listdir
from os.path import join

import numpy as np
import scipy.ndimage
import six
from deep_folding.brainvisa.load_data import fetch_data
from deep_folding.brainvisa.utils import remove_hull
from deep_folding.brainvisa.utils.bbox import compute_max_box
from deep_folding.brainvisa.utils.logs import LogJson
from deep_folding.brainvisa.utils.logs import log_command_line
from deep_folding.brainvisa.utils.mask import compute_centered_mask
from deep_folding.brainvisa.utils.mask import compute_simple_mask
from deep_folding.brainvisa.utils.resample import resample
from deep_folding.brainvisa.utils.sulcus_side import complete_sulci_name
from joblib import cpu_count
from pqdm.processes import pqdm
from soma import aims
from tqdm import tqdm

_ALL_SUBJECTS = -1

_SIDE_DEFAULT = 'L'  # hemisphere 'L' or 'R'

_CROPPING_DEFAULT = 'mask'  # crops according to a mask by default

_OUT_VOXEL_SIZE = (1, 1, 1)  # default output voxel size

_EXTERNAL = 11  # topological value meaning "outside the brain"

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'S.T.s.ter.asc.ant.'

_COMBINE_TYPE = False

_DIST_MAP_DEFAULT = False

# Input directories
# -----------------

# Input directory contaning the skeletons and labels
_SRC_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/datasets/hcp'

# Input directory contaning the morphologist analysis of the HCP database
_GRAPH_DIR_DEFAULT = '/neurospin/hcp'

# Directory where subjects to be processed are stored.
# Default is for HCP dataset
_MORPHOLOGIST_DIR_DEFAULT = 'ANALYSIS/3T_morphologist'

# Directory containing bounding box json files
# default corresponds to bounding boxes computed for voxels of 1mm
_BBOX_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/current/bbox'

# Directory containing mask files
_MASK_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/current/mask'

# Directory containing bounding box json files
# default corresponds to bounding boxes computed for voxinput
# -------------------------
_TGT_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/test'

_VERBOSE_DEFAULT = False

# temporary directory
temp_dir = tempfile.mkdtemp()


def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus - 2, 1)


class DatasetCroppedSkeleton:
    """Generates cropped skeleton files and corresponding pickle file
    """

    def __init__(self,
                 graph_dir=_GRAPH_DIR_DEFAULT,
                 src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 morphologist_dir=_MORPHOLOGIST_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 cropping=_CROPPING_DEFAULT,
                 out_voxel_size=_OUT_VOXEL_SIZE,
                 combine_type=_COMBINE_TYPE,
                 dist_map=_DIST_MAP_DEFAULT):
        """Inits with list of directories and list of sulci

        Args:
            graph_dir: list of strings naming full path source directories,
                    containing MRI and graph images
            src_dir: folder containing generated skeletons and labels
            tgt_dir: name of target (output) directory with full path
            transform_dir: directory containing transformation files
                    (generated using transform.py)
            bbox_dir: directory containing bbox json files
                    (generated using bounding_box.py)
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left, or R for right hemisphere)
        """

        self.graph_dir = graph_dir
        self.src_dir = src_dir
        self.side = side
        # Transforms sulcus in a list of sulci
        self.list_sulci = ([list_sulci] if isinstance(list_sulci, str)
                           else list_sulci)
        self.list_sulci = complete_sulci_name(self.list_sulci, self.side)
        self.tgt_dir = tgt_dir
        self.bbox_dir = bbox_dir
        self.mask_dir = mask_dir
        self.morphologist_dir = morphologist_dir
        self.cropping = cropping
        self.out_voxel_size = out_voxel_size
        self.combine_type = combine_type
        self.dist_map = dist_map

        # Morphologist directory
        self.morphologist_dir = join(self.graph_dir, self.morphologist_dir)

        # default acquisition subdirectory
        self.acquisition_dir = "%(subject)s/t1mri/default_acquisition"

        # Directory where to store cropped skeleton files
        self.cropped_skeleton_dir = join(self.tgt_dir, self.side + 'crops')

        # Directory where to store cropped label files
        self.cropped_label_dir = join(self.tgt_dir, self.side + 'labels')

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        # Files from morphologist pipeline
        #self.skeleton_file = 'default_analysis/segmentation/' \
        #                    '%(side)sskeleton_%(subject)s.nii.gz'
        ## FOR HCP dataset
        # self.skeleton_file = '/neurospin/dico/data/deep_folding/datasets/hcp/' \
        #                            '%(side)sskeleton_%(subject)s_generated.nii.gz'
        self.distMap_file = '/neurospin/dico/data/deep_folding/datasets/hcp/distance_map/R/' \
                                   'distance_map_%(subject)s.nii.gz'
        ## FOR TISSIER dataset
        # self.skeleton_file = '/neurospin/dico/data/deep_folding/datasets/ACC_patterns/tissier/' \
        #                             '%(side)sskeleton_%(subject)s_generated.nii.gz'
        self.graph_file = 'default_analysis/folds/3.1/default_session_auto/' \
            '%(side)s%(subject)s_default_session_auto.arg'

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.cropped_skeleton_file = '%(subject)s_cropped_skeleton.nii.gz'
        self.cropped_label_file = '%(subject)s_cropped_label.nii.gz'

        # Initialization of bounding box coordinates
        self.bbmin = np.zeros(3)
        self.bbmax = np.zeros(3)

        # Creates json log class
        json_file = join(self.tgt_dir, self.side + 'dataset.json')
        self.json = LogJson(json_file)

        # reference file in MNI template with corrct voxel size
        self.ref_file = f"{temp_dir}/file_ref.nii.gz"
        self.g_to_icbm_template_file = join(
            temp_dir, 'file_g_to_icbm_%(subject)s.trm')

    def define_referentials(self):
        """Writes MNI 2009 reference file with output voxel size

        It will be used by AimsApplyTransform
        """
        hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
        voxel_size = np.concatenate((self.out_voxel_size, [1]))
        resampling_ratio = np.array(hdr['voxel_size']) / voxel_size

        orig_dim = hdr['volume_dimension']
        new_dim = list((resampling_ratio * orig_dim).astype(int))

        vol = aims.Volume(new_dim, dtype='S16')
        vol.copyHeaderFrom(hdr)
        vol.header()['voxel_size'] = voxel_size
        aims.write(vol, self.ref_file)

    def crop_bbox(self, file_cropped, verbose):
        """Crops according to bounding box"""
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
        if verbose:
            os.popen(cmd_crop).read()
        else:
            var_output = os.popen(cmd_crop).read()

    def filter_mask(self):
        """Smooths the mask with Gaussian Filter
        """
        arr = np.asarray(self.mask)
        arr_filter = scipy.ndimage.gaussian_filter(
            arr.astype(float),
            sigma=0.5,
            order=0,
            output=None,
            mode='reflect',
            truncate=4.0)
        arr[:] = (arr_filter > 0.001).astype(int)

    def crop_mask(self, file_cropped, verbose):
        """Crops according to mask"""
        vol = aims.read(file_cropped)

        arr = np.asarray(vol)
        # remove_hull.remove_hull(arr)

        arr_mask = np.asarray(self.mask)
        arr[arr_mask == 0] = 0
        arr[arr == _EXTERNAL] = 0

        # Take the coordinates of the bounding box
        bbmin = self.bbmin
        bbmax = self.bbmax
        xmin, ymin, zmin = str(bbmin[0]), str(bbmin[1]), str(bbmin[2])
        xmax, ymax, zmax = str(bbmax[0]), str(bbmax[1]), str(bbmax[2])

        aims.write(vol, file_cropped)

        # Defines rop of the images based on bounding box
        cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + \
            ' -X ' + xmax + ' -Y ' + ymax + ' -Z ' + zmax
        cmd_crop = 'AimsSubVolume' + \
            ' -i ' + file_cropped + \
            ' -o ' + file_cropped + cmd_bounding_box

        if verbose:
            os.popen(cmd_crop).read()
        else:
            var_output = os.popen(cmd_crop).read()

    def crop_one_file(self, subject_id, verbose=False):
        """Crops one file

        Args:
            subject_id: string giving the subject ID
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        subject = {'subject': subject_id, 'side': self.side}
        # FOR TISSIER
        # subject_id = re.search('([ae\d]{5,6})', subject_id).group(0)

        # Names directory where subject analysis files are stored
        subject_dir = \
            join(self.morphologist_dir, self.acquisition_dir % subject)

        if self.dist_map:
            target_file = join(subject_dir, self.distMap_file % {'subject': subject_id})
            print(target_file)
        else:
            # Skeleton file name
            target_file = join(subject_dir, self.skeleton_file % {'subject': subject_id, 'side': self.side})

        # Creates transformation MNI template
        file_graph = join(subject_dir, self.graph_file % subject)
        graph = aims.read(file_graph)
        g_to_icbm_template = aims.GraphManip.getICBM2009cTemplateTransform(
            graph)
        g_to_icbm_template_file = self.g_to_icbm_template_file % subject
        aims.write(g_to_icbm_template, g_to_icbm_template_file)

        if os.path.exists(target_file):
            # Creates output (cropped) file name
            file_cropped_skeleton = join(
                self.cropped_skeleton_dir,
                self.cropped_skeleton_file % {
                    'subject': subject_id,
                    'side': self.side})

            # We give values with ascendent priority
            # The more important is the inversion in the priority
            # for the bottom value (30) and the simple surface value (60)
            # with respect to the natural order
            # We don't give background
            values = np.array([90, 80, 70, 50, 40, 20, 10, 30, 60, 11])

            # Normalization and resampling of skeleton images
            if self.resampling:
                resampled = resample(input_image=target_file,
                                     output_vs=self.out_voxel_size,
                                     transformation=g_to_icbm_template_file,
                                     verbose=False)
                aims.write(resampled, file_cropped)
            else :
                cmd_normalize = 'AimsApplyTransform' + \
                                ' -i ' + target_file + \
                                ' -o ' + file_cropped + \
                                ' -m ' + g_to_icbm_template_file + \
                                ' -r ' + self.ref_file + \
                                ' -t ' + self.interp
                os.system(cmd_normalize)
                print(cmd_normalize)

            # Cropping of skeleton image
            if self.cropping == 'bbox':
                self.crop_bbox(file_cropped_skeleton, verbose)
            else:
                self.crop_mask(file_cropped_skeleton, verbose)
        else:
            raise FileNotFoundError(f"{file_skeleton} not found")

        if os.path.exists(file_foldlabel):
            # Creates output (cropped) file name
            file_cropped_label = join(
                self.cropped_label_dir,
                self.cropped_label_file % {
                    'subject': subject_id,
                    'side': self.side})

            # Normalization and resampling of skeleton images
            resampled = resample(input_image=file_foldlabel,
                                 output_vs=self.out_voxel_size,
                                 transformation=g_to_icbm_template_file,
                                 verbose=False)
            aims.write(resampled, file_cropped_label)

            # Cropping of skeleton image
            if self.cropping == 'bbox':
                self.crop_bbox(file_cropped_label, verbose)
            else:
                self.crop_mask(file_cropped_label, verbose)
        else:
            raise FileNotFoundError(f"{file_foldlabel} not found")

    def crop_files(self, number_subjects=_ALL_SUBJECTS):
        """Crop nii files

        The programm loops over all subjects from the input (source) directory.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if number_subjects:

            # subjects are detected as the nifti file names under src_dir
            expr = '^.skeleton_generated_([0-9a-zA-Z]*).nii.gz$'
            if os.path.isdir(self.skeleton_dir):
                list_all_subjects = [re.search(expr, os.path.basename(dI))[1]
                                     for dI in glob.glob(f"{self.skeleton_dir}/*.nii.gz")]
            else:
                raise NotADirectoryError(
                    f"{self.sksleton_dir} doesn't exist or is not a directory")

            # Gives the possibility to list only the first number_subjects
            list_subjects = (
                list_all_subjects
                if number_subjects == _ALL_SUBJECTS
                else list_all_subjects[:number_subjects])

            # Creates target and cropped directory
            if not os.path.exists(self.tgt_dir):
                os.makedirs(self.tgt_dir)
            if not os.path.exists(self.cropped_skeleton_dir):
                os.makedirs(self.cropped_skeleton_dir)
            if not os.path.exists(self.cropped_label_dir):
                os.makedirs(self.cropped_label_dir)

            # Writes number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(list_subjects),
                           'graph_dir': self.graph_dir,
                           'src_dir': self.src_dir,
                           'bbox_dir': self.bbox_dir,
                           'mask_dir': self.mask_dir,
                           'side': self.side,
                           'list_sulci': self.list_sulci,
                           'bbmin': self.bbmin.tolist(),
                           'bbmax': self.bbmax.tolist(),
                           'tgt_dir': self.tgt_dir,
                           'cropped_skeleton_dir': self.cropped_skeleton_dir,
                           'cropped_label_dir': self.cropped_label_dir,
                           'resampling_type': 'sulcus-based',
                           'out_voxel_size': self.out_voxel_size,
                           'combine_type': self.combine_type,
                           'dist_map': self.dist_map
                           }
            self.json.update(dict_to_add=dict_to_add)

            # Defines referential
            self.define_referentials()

            # Performs cropping for each file in a parallelized way
            print("list_subjects = ", list_subjects)

            for sub in list_subjects:
                 self.crop_one_file(sub)
            #pqdm(list_subjects, self.crop_one_file, n_jobs=define_njobs())

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

        # Computes bounding box and mask
        if number_subjects:
            if self.cropping == 'bbox':
                self.bbmin, self.bbmax = compute_max_box(sulci_list=self.list_sulci,
                                                         side=self.side,
                                                         talairach_box=False,
                                                         src_dir=self.bbox_dir)
            elif self.cropping == 'mask':
                if self.combine_type:
                    self.mask, self.bbmin, self.bbmax = \
                        compute_centered_mask(sulci_list=self.list_sulci,
                                              side=self.side,
                                              mask_dir=self.mask_dir)
                else:
                    self.mask, self.bbmin, self.bbmax = \
                        compute_simple_mask(sulci_list=self.list_sulci,
                                            side=self.side,
                                            mask_dir=self.mask_dir)
            else:
                raise ValueError(
                    'Cropping must be either \'bbox\' or \'mask\'')

        # Generate cropped files
        self.crop_files(number_subjects=number_subjects)

        # Creation of .pickle file for all subjects
        if number_subjects:
            fetch_data(cropped_dir=self.cropped_skeleton_dir,
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
        "-g", "--graph_dir", type=str, default=_GRAPH_DIR_DEFAULT,
        help='Source directory where the graph lies. '
             'Default is : ' + _GRAPH_DIR_DEFAULT)
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where skeletons and labels lie. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the cropped and pickle files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-a", "--mask_dir", type=str, default=_MASK_DIR_DEFAULT,
        help='masking directory where mask has been stored. '
             'Default is : ' + _MASK_DIR_DEFAULT)
    parser.add_argument(
        "-b", "--bbox_dir", type=str, default=_BBOX_DIR_DEFAULT,
        help='Bounding box directory where json files containing '
             'bounding box coordinates have been stored. '
             'Default is : ' + _BBOX_DIR_DEFAULT)
    parser.add_argument(
        "-m",
        "--morphologist_dir",
        type=str,
        default=_MORPHOLOGIST_DIR_DEFAULT,
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
        "-c", "--cropping", type=str, default=_CROPPING_DEFAULT,
        help='Method of to select and crop the image. '
             'Type of cropping: '
             'bbox: for bounding box cropping'
             'mask: selection based on a mask'
             'Default is : mask')
    parser.add_argument(
        "-x",
        "--out_voxel_size",
        type=float,
        nargs='+',
        default=_OUT_VOXEL_SIZE,
        help='Voxel size of output images'
        'Default is : 1 1 1')
    parser.add_argument(
        "-v", "--verbose",
        default=False,
        action='store_true',
        help='If verbose is true, no parallelism.')
    parser.add_argument(
        "-o", "--combine_type", type=bool, default=_COMBINE_TYPE,
        help='Whether use specific combination of masks or not')
    parser.add_argument(
        "-d", "--dist_map", type=bool, default=_DIST_MAP_DEFAULT,
        help='Whether crop and normalize distance map instead of skeleton')

    params = {}

    args = parser.parse_args(argv)

    # Writes command line argument to target dir for logging
    log_command_line(args, "dataset_gen_pipe.py", args.tgt_dir)

    params['src_dir'] = args.src_dir
    params['graph_dir'] = args.graph_dir
    params['tgt_dir'] = args.tgt_dir
    params['bbox_dir'] = args.bbox_dir
    params['mask_dir'] = args.mask_dir
    params['list_sulci'] = args.sulcus  # a list of sulci
    params['side'] = args.side
    params['cropping'] = args.cropping
    params['out_voxel_size'] = tuple(args.out_voxel_size)
    params['morphologist_dir'] = args.morphologist_dir
    params['combine_type'] = args.combine_type
    params['dist_map'] = args.dist_map

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


def dataset_gen_pipe(graph_dir=_GRAPH_DIR_DEFAULT,
                     src_dir=_SRC_DIR_DEFAULT,
                     tgt_dir=_TGT_DIR_DEFAULT,
                     bbox_dir=_BBOX_DIR_DEFAULT,
                     mask_dir=_MASK_DIR_DEFAULT,
                     morphologist_dir=_MORPHOLOGIST_DIR_DEFAULT,
                     side=_SIDE_DEFAULT,
                     list_sulci=_SULCUS_DEFAULT,
                     number_subjects=_ALL_SUBJECTS,
                     cropping=_CROPPING_DEFAULT,
                     out_voxel_size=_OUT_VOXEL_SIZE,
                     combine_type=_COMBINE_TYPE,
                     dist_map=_DIST_MAP_DEFAULT):
    """Main program generating cropped files and corresponding pickle file
    """

    dataset = DatasetCroppedSkeleton(graph_dir=graph_dir,
                                     src_dir=src_dir,
                                     tgt_dir=tgt_dir,
                                     bbox_dir=bbox_dir,
                                     mask_dir=mask_dir,
                                     morphologist_dir=morphologist_dir,
                                     side=side,
                                     list_sulci=list_sulci,
                                     cropping=cropping,
                                     out_voxel_size=out_voxel_size,
                                     combine_type=combine_type,
                                     dist_map=dist_map)
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
        dataset_gen_pipe(graph_dir=params['graph_dir'],
                         src_dir=params['src_dir'],
                         tgt_dir=params['tgt_dir'],
                         bbox_dir=params['bbox_dir'],
                         mask_dir=params['mask_dir'],
                         morphologist_dir=params['morphologist_dir'],
                         side=params['side'],
                         list_sulci=params['list_sulci'],
                         number_subjects=params['nb_subjects'],
                         cropping=params['cropping'],
                         out_voxel_size=params['out_voxel_size'],
                         combine_type=params['combine_type'],
                         dist_map=params['dist_map'])
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