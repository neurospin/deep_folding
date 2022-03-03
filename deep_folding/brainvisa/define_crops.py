#!python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
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
The aim of this script is to output bounding box got given sulci
based on a manually labelled dataset

Bounding box corresponds to the biggest box that encompasses the given sulci
on all subjects of the manually labelled dataset. It measures the bounding box
in the MNI152  space
"""

#from __future__ import division
#from __future__ import print_function

import sys
import glob
import os
from os.path import join
import argparse
import six

import numpy as np
import scipy.ndimage

from soma import aims
from deep_folding.anatomist_tools.utils.logs import LogJson
from deep_folding.anatomist_tools.utils.resample import resample
from deep_folding.anatomist_tools.utils.bbox import compute_max
from deep_folding.anatomist_tools.utils.sulcus_side import complete_sulci_name
from deep_folding.anatomist_tools.utils.logs import log_command_line

_ALL_SUBJECTS = -1

# Default directory in which lies the manually segmented database
_SRC_DIR_DEFAULT = "/neurospin/dico/data/bv_databases/human/pclean/all"

# Default directory to which we write the bounding box results
_bbox_dir_DEFAULT = "/neurospin/dico/data/deep_folding/test/bbox"

# Default directory to which we write the masks
_MASK_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/mask"

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'L'

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'S.T.s.ter.asc.ant.'

# Gives the relative path to the manually labelled graph .arg
# in the supervise
_PATH_TO_GRAPH_DEFAULT = "t1mri/t1/default_analysis/folds/3.3/base2018_manual"


class BoundingBoxMax:
    """Determines the maximum Bounding Box around given sulci

    It is determined in the  MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                 bbox_dir=_bbox_dir_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 out_voxel_size=None):
        """Inits with list of directories and list of sulci

        Args:
            src_dir: list of strings naming ful path source directories
            path_to_graph: list of strings naming relative path to labelled graph
            bbox_dir: name of target directory with full path
            sulcus: sulcus name
            side: hemisphere side (either L for left, or R for right hemisphere)
        """

        # Transforms input source dir to a list of strings
        self.src_dir = [src_dir] if isinstance(src_dir, str) else src_dir

        # manually labelled graph file relative to the subject directory
        # we use the '*' glob to take into account different naming conventions
        # It must be put in the same order as src_dir
        path_to_graph = ([path_to_graph] if isinstance(path_to_graph, str)
                         else path_to_graph)
        self.graph_file = []
        for path in path_to_graph:
            self.graph_file.append('%(subject)s/' \
                              + path \
                              + '/%(side)s%(subject)s*.arg')

        self.sulcus = sulcus
        self.bbox_dir = bbox_dir
        self.mask_dir = mask_dir
        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.voxel_size_out = (out_voxel_size,
                               out_voxel_size,
                               out_voxel_size,
                               1)

        # Json full name is the name of the sulcus + .json
        # and is kept under the subdirectory Left or Right
        json_file = join(self.bbox_dir, self.side, self.sulcus + '.json')
        self.json = LogJson(json_file)
        self.mask = aims.Volume()
        self.mask_file = join(self.mask_dir, self.side, self.sulcus + '.nii.gz')

    def list_all_subjects(self):
        """List all subjects from the clean database (directory src_dir).

        Subjects are the names of the subdirectories of the root directory.

        Parameters:

        Returns:
            subjects: a list containing all subjects to be analyzed
        """

        subjects = []

        # Main loop: list all subjects of the directories
        # listed in self.src_dir
        for src_dir, graph_file in zip(self.src_dir, self.graph_file):
            for filename in os.listdir(src_dir):
                directory = os.path.join(src_dir, filename)
                if os.path.isdir(directory):
                    if filename != 'ra':
                        subject = filename
                        subject_d = {'subject': subject,
                                     'side': self.side,
                                     'dir': src_dir,
                                     'graph_file': graph_file % {'side': self.side, 'subject': subject}}
                        subjects.append(subject_d)

        return subjects

    def create_mask(self):
        """Creates aims volume in MNI ICBM152 nonlinear 2009c asymmetrical template
        http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip

        """

        # Creates and puts to 0 an aims volume
        # with the correct size and pixel size
        hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
        resampling_ratio = np.array(hdr['voxel_size']) / self.voxel_size_out
        orig_dim = hdr['volume_dimension']
        new_dim = list((resampling_ratio * orig_dim).astype(int))

        self.mask = aims.Volume(new_dim, dtype='S16')
        self.mask.copyHeaderFrom(hdr)
        self.mask.header()['voxel_size'] = self.voxel_size_out

    def increment_one_mask(self, graph_filename):
        """Increments self.mask of 1 where there is the sulcus

        Parameters:
            graph_filename: string being the name of graph file .arg to analyze:
                            for example: 'Lammon_base2018_manual.arg'

        """

        # Reads the data graph and transforms it to MNI ICBM152 referential
        graph = aims.read(graph_filename)
        g_to_icbm_template = aims.GraphManip.getICBM2009cTemplateTransform(graph)
        voxel_size_in = graph['voxel_size'][:3]
        arr = np.asarray(self.mask)

        # Gets the min and max coordinates of the sulci
        # by looping over all the vertices of the graph
        for vertex in graph.vertices():
            vname = vertex.get('name')
            if vname != self.sulcus:
                continue
            for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
                bucket = vertex.get(bucket_name)
                if bucket is not None:
                    voxels_real = np.asarray(
                        [g_to_icbm_template.transform(np.array(voxel) * voxel_size_in)
                         for voxel in bucket[0].keys()])
                    if voxels_real.shape == (0,):
                        continue
                    voxels = np.round(np.array(voxels_real) / self.voxel_size_out[:3]).astype(int)

                    if voxels.shape == (0,):
                        continue
                    for i,j,k in voxels:
                        arr[i,j,k,0] += 1

    def get_one_bounding_box(self, graph_filename):
        """get bounding box of the chosen sulcus for one data graph in MNI 152

      Function that outputs the bounding box for the listed sulci
      for this datagraph. The bounding box is the smallest rectangular box
      that encompasses the chosen sulcus.
      It is given in the MNI 152 referential.

      Parameters:
        graph_filename: string being the name of graph file .arg to analyze:
                        for example: 'Lammon_base2018_manual.arg'

      Returns:
        bbox_min: numpy array giving the upper right vertex coordinates
                of the box in the MNI 152 referential
        bbox_max: numpy array fiving the lower left vertex coordinates
                of the box in the MNI 152 referential
      """

        # Reads the data graph and transforms it to AIMS Talairach referential
        # Note that this is NOT the MNI Talairach referential
        # This is the Talairach referential used in AIMS
        # There are several Talairach referentials
        graph = aims.read(graph_filename)
        voxel_size_in = graph['voxel_size'][:3]
        g_to_icbm_template = \
            aims.GraphManip.getICBM2009cTemplateTransform(graph)
        bbox_min = None
        bbox_max = None

        # Gets the min and max coordinates of the sulci
        # by looping over all the vertices of the graph
        for vertex in graph.vertices():
            vname = vertex.get('name')
            if vname != self.sulcus:
                continue
            for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
                bucket = vertex.get(bucket_name)
                if bucket is not None:
                    voxels = np.asarray(
                        [g_to_icbm_template.transform(np.array(voxel) * voxel_size_in)
                         for voxel in bucket[0].keys()])
                    if voxels.shape == (0,):
                        continue

                    bbox_min = np.min(np.vstack(
                        ([bbox_min] if bbox_min is not None else [])
                        + [voxels]), axis=0)
                    bbox_max = np.max(np.vstack(
                        ([bbox_max] if bbox_max is not None else [])
                        + [voxels]), axis=0)

        print('box (MNI 152) min:', bbox_min)
        print('box (MNI 152) max:', bbox_max)

        return bbox_min, bbox_max

    def get_bounding_boxes(self, subjects):
        """get bounding boxes of the chosen sulcus for all subjects.

      Function that outputs the bounding box for the listed sulci on a manually
      labeled dataset.
      Bounding box corresponds to the biggest box encountered in the manually
      labeled subjects in the MNI1 152 space.
      The bounding box is the smallest rectangular box that
      encompasses the sulcus.

      Parameters:
        subjects: list containing all subjects to be analyzed

      Returns:
        list_bbmin: list containing the upper right vertex of the box
                    in the MNI 152 space
        list_bbmax: list containing the lower left vertex of the box
                    in the MNI 152 space
      """

        # Initialization
        list_bbmin = []
        list_bbmax = []

        for sub in subjects:
            print(sub)
            # It substitutes 'subject' in graph_file name
            graph_file = sub['graph_file'] % sub
            # It looks for a graph file .arg
            sulci_pattern = glob.glob(join(sub['dir'], graph_file))[0]

            bbox_min, bbox_max = \
                self.get_one_bounding_box(sulci_pattern % sub)
            if bbox_min is not None:
                list_bbmin.append([bbox_min[0], bbox_min[1], bbox_min[2]])
                list_bbmax.append([bbox_max[0], bbox_max[1], bbox_max[2]])
            else:
                print(f"No sulcus {self.sulcus} found for {sub}; it can be OK.")

        if not list_bbmin:
            raise ValueError(f"No sulcus named {self.sulcus} found "
                        'for the whole dataset. '
                        'It is an error. You should check sulcus name.')

        return list_bbmin, list_bbmax

    def increment_mask(self, subjects):
        """increment mask for the chosen sulcus for all subjects

        Parameters:
            subjects: list containing all subjects to be analyzed
        """

        for sub in subjects:
            print(sub)
            # It substitutes 'subject' in graph_file name
            graph_file = sub['graph_file'] % sub
            # It looks for a graph file .arg
            sulci_pattern = glob.glob(join(sub['dir'], graph_file))[0]

            self.increment_one_mask(sulci_pattern % sub)
        # self.mask /= float(len(subjects))

    def write_mask(self):
        """Writes mask on mask file"""
        mask_file_dir = os.path.dirname(self.mask_file)
        os.makedirs(mask_file_dir, exist_ok=True)
        print(self.mask_file)
        aims.write(self.mask, self.mask_file)

    def compute_box_voxel(self, bbmin_mni152, bbmax_mni152):
        """Returns the coordinates of the box as voxels

      Coordinates of the box in voxels are determined in the MNI referential

      Parameters:
        bbmin_mni152: numpy array with the coordinates of the upper right corner
                of the box (MNI152 space)
        bbmax_mni152: numpy array with the coordinates of the lower left corner
                of the box (MNI152 space)
        voxel_size: voxel size (in MNI referential or HCP normalized SPM space)

      Returns:
        bbmin_vox: numpy array with the coordinates of the upper right corner
                of the box (voxels in MNI space)
        bbmax_vox: numpy array with the coordinates of the lower left corner
                of the box (voxels in MNI space)
      """

        # To go back from mms to voxels
        voxel_size = self.voxel_size_out
        bbmin_vox = np.round(np.array(bbmin_mni152) / voxel_size[:3]).astype(int)
        bbmax_vox = np.round(np.array(bbmax_mni152) / voxel_size[:3]).astype(int)

        return bbmin_vox, bbmax_vox

    def transform_to_aims_talairach(self, bbmin_mni152, bbmax_mni152):
        """Transform bbox coordinates from MNI152 to AIMS talairach referential"""

        g_icbm_template_to_talairach = \
            aims.StandardReferentials.talairachToICBM2009cTemplate().inverse()
        bbmin_tal = g_icbm_template_to_talairach.transform(bbmin_mni152)
        bbmin_tal = np.asarray(bbmin_tal)
        bbmax_tal = g_icbm_template_to_talairach.transform(bbmax_mni152)
        bbmax_tal = np.asarray(bbmax_tal)
        print('box (AIMS Talairach) min:', bbmin_tal.tolist())
        print('box (AIMS Talairach) max:', bbmax_tal.tolist())

        return bbmin_tal, bbmax_tal

    def compute_bounding_box(self, number_subjects=_ALL_SUBJECTS):
        """Main class program to compute the bounding box

        Args:
            number_subjects: number_subjects to analyze
        """

        if number_subjects:
            subjects = self.list_all_subjects()

            self.json.write_general_info()

            # Gives the possibility to list only the first number_subjects
            subjects = (
                subjects
                if number_subjects == _ALL_SUBJECTS
                else subjects[:number_subjects])

            # Creates target bbox dir if it doesn't exist
            if not os.path.exists(self.bbox_dir):
                os.makedirs(self.bbox_dir)

            # Creates target mask dir if it doesn't exist
            if not os.path.exists(self.mask_dir):
                os.makedirs(self.mask_dir)

            # Writes number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(subjects),
                           'src_dir': self.src_dir,
                           'bbox_dir': self.bbox_dir,
                           'out_voxel_size': self.voxel_size_out[0]}
            self.json.update(dict_to_add=dict_to_add)

            # Creates volume that will take the mask
            self.create_mask()

            # Increments mask for each sulcus and subjects
            self.increment_mask(subjects)

            # Smoothing and filling of the mask with gaussian filtering
            #self.filter_mask()

            # Saving of generated masks
            self.write_mask()

            # Determines the box encompassing the sulcus for all subjects
            # The coordinates are determined in MNI 152  space
            list_bbmin, list_bbmax = self.get_bounding_boxes(subjects)
            bbmin_mni152, bbmax_mni152 = compute_max(list_bbmin, list_bbmax)

            bbmin_tal, bbmax_tal = \
                self.transform_to_aims_talairach(bbmin_mni152, bbmax_mni152)

            dict_to_add = {'bbmin_MNI152': bbmin_mni152.tolist(),
                           'bbmax_MNI152': bbmax_mni152.tolist(),
                           'bbmin_AIMS_Talairach': bbmin_tal.tolist(),
                           'bbmax_AIMS_Talairach': bbmax_tal.tolist()}

            # Determines the box encompassing the sulcus for all subjects
            # The coordinates are determined in voxels in MNI space
            bbmin_vox, bbmax_vox = self.compute_box_voxel(bbmin_mni152,
                                                          bbmax_mni152)

            dict_to_add.update({'side': self.side,
                                'sulcus': self.sulcus,
                                'bbmin_voxel': bbmin_vox.tolist(),
                                'bbmax_voxel': bbmax_vox.tolist()})
            self.json.update(dict_to_add=dict_to_add)
            print("box (voxel): min = ", bbmin_vox)
            print("box (voxel): max = ", bbmax_vox)

        else:
            bbmin_vox = 0
            bbmax_vox = 0

        return bbmin_vox, bbmax_vox


def bounding_box(src_dir=_SRC_DIR_DEFAULT,
                 bbox_dir=_bbox_dir_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                 sulcus=_SULCUS_DEFAULT, side=_SIDE_DEFAULT,
                 number_subjects=_ALL_SUBJECTS,
                 out_voxel_size=None):
    """ Main program computing the box encompassing the sulcus in all subjects

  The programm loops over all subjects
  and computes in MNI space the voxel coordinates of the box encompassing
  the sulci for all subjects

  Args:
      src_dir: list of strings -> directories of the supervised databases
      bbox_dir: string giving target bbox directory path
      mask_dir: string giving target mask directory path
      path_to_graph: string giving relative path to manually labelled graph
      side: hemisphere side (either 'L' for left, or 'R' for right)
      sulcus: string giving the sulcus to analyze
      number_subjects: integer giving the number of subjects to analyze,
            by default it is set to _ALL_SUBJECTS (-1)
      skeleton_file: skeleton file of the reference subject
  """

    box = BoundingBoxMax(src_dir=src_dir, bbox_dir=bbox_dir,
                         mask_dir=mask_dir,
                         path_to_graph=path_to_graph,
                         sulcus=sulcus, side=side,
                         out_voxel_size=out_voxel_size)
    bbmin_vox, bbmax_vox = box.compute_bounding_box(
        number_subjects=number_subjects)

    return bbmin_vox, bbmax_vox, box.mask


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: a dictionary with all arugments as keys
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='define_crops.py',
        description='Computes mask and bounding box around the named sulcus')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT, nargs='+',
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--bbox_dir", type=str, default=_bbox_dir_DEFAULT,
        help='Target directory where to store the output bbox json files. '
             'Default is : ' + _bbox_dir_DEFAULT)
    parser.add_argument(
        "-m", "--mask_dir", type=str, default=_MASK_DIR_DEFAULT,
        help='Target directory where to store the output mask files. '
             'Default is : ' + _MASK_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box. '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-p", "--path_to_graph", type=str,
        default=_PATH_TO_GRAPH_DEFAULT,
        help='Relative path to manually labelled graph. '
             'Default is ' + _PATH_TO_GRAPH_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose. '
             'Default is : all')
    parser.add_argument(
        "-v", "--out_voxel_size", type=float, default=None,
        help='Voxel size of of bounding box. '
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)

    # Writes command line argument to target dir for logging
    log_command_line(args, "generate_skeleton.py", args.tgt_dir)

    params['src_dir'] = args.src_dir  # src_dir is a list
    params['path_to_graph'] = args.path_to_graph
    params['bbox_dir']= args.bbox_dir # bbox_dir is a string, only one directory
    params['mask_dir']= args.mask_dir # bbox_dir is a string, only one directory
    params['sulcus'] = args.sulcus  # sulcus is a string
    params['side'] = args.side
    params['out_voxel_size'] = args.out_voxel_size

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
        raise ValueError("nb_subjects must be either the string \"all\" "
                         "or an integer")
    params['nb_subjects'] = number_subjects

    return params


def main(argv):
    """Reads argument line and determines the max bounding box

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        params = parse_args(argv)
        # Actual API
        bounding_box(src_dir=params['src_dir'],
                     path_to_graph=params['path_to_graph'],
                     bbox_dir=params['bbox_dir'],
                     mask_dir=params['mask_dir'],
                     sulcus=params['sulcus'],
                     side=params['side'],
                     number_subjects=params['nb_subjects'],
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
