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

"""
The aim of this script is to output bounding box got given sulcis
based on a manually labelled dataset

Bounding box corresponds to the biggest box that encompasses the given sulci
on all subjects of the manually labelled dataset. It measures the bounding box
in the normalized SPM space
"""

from __future__ import division
from __future__ import print_function

import sys

from soma import aims
import os
import numpy as np
import argparse
import six

_ALL_SUBJECTS = -1

# Default directory in which lies the manually segmented database
_SRC_DIR_DEFAULT = "/neurospin/lnao/PClean/database_learnclean/all/"

# Default directory to which we write the bounding box results
_TGT_DIR_DEFAULT = "/neurospin/dico/deep_folding_data/data/bbox"

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'L'

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'S.T.s.ter.asc.ant._left'

######################################################################
# Global variables (that the user will probably not change)
######################################################################

# A normalized SPM image to get the HCP morphologist transformation
_image_normalized_SPM = '/neurospin/hcp/ANALYSIS/3T_morphologist/100206/t1mri/default_acquisition/normalized_SPM_100206.nii'


class BoundingBoxMax:
    """Determines the maximum Bounding Box around given sulci

    Attributes:

    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT):
        """Inits with list of directories and list of sulci

        Args:
            src_dir: list of strings naming src directories
            sulcus: list of sulcus names
        """

        # Transforms input sourcedir and sulcus names to a list of strings
        self.src_dir = [src_dir] if type(src_dir) is str else src_dir
        self.sulcus = [sulcus] if type(sulcus) is str else sulcus

        self.tgt_dir = tgt_dir
        self.side = side

        # Creates json file name
        self.json_file = join(self.tgt_dir, 'bbox.json')
        self.create_json()

    def list_all_subjects(self):
        """List all subjects from the clean database (directory _root_dir).

        Subjects are the names of the subdirectories of the root directory.

        Parameters:

        Returns:
            subjects: a list containing all subjects to be analyzed
        """

        subjects = []

        # Main loop: list all subjects of the directories
        # listed in self.src_dir
        for src_dir in self.src_dir:
            for filename in os.listdir(src_dir):
                directory = os.path.join(src_dir, filename)
                if os.path.isdir(directory):
                    if filename != 'ra':
                        subject_d = {'subject': filename,
                                     'side': self.side,
                                     'dir': src_dir}
                        subjects.append(subject_d)

        return subjects

    def get_one_bounding_box(self, graph_filename):
        """get bounding box of the chosen sulcus for one data graph

      Function that outputs the bounding box for the listed sulci
      for this datagraph. The bounding box is the smallest rectangular box
      that encompasses the chosen sulcus.

      Parameters:
        graph_filename: string being the name of graph file .arg to analyze:
                        for example: 'Lammon_base2018_manual.arg'

      Returns:
        bbox_min: numpy array giving the upper right vertex coordinates
                of the box in the Talairach space
        bbox_max: numpy array fiving the lower left vertex coordinates
                of the box in the Talairach space
      """

        # Read the data graph and extract the Talairach transform
        graph = aims.read(graph_filename)
        voxel_size = graph['voxel_size'][:3]
        tal_transfo = aims.GraphManip.talairach(graph)
        bbox_min = None
        bbox_max = None

        # Get the min and max coordinates of the sulcus '_sulcus'
        # by looping over all the vertices of the graph
        for vertex in graph.vertices():
            vname = vertex.get('name')
            if vname not in self.sulcus:
                continue
            for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
                bucket = vertex.get(bucket_name)
                if bucket is not None:
                    voxels = np.asarray(
                        [tal_transfo.transform(np.array(voxel) * voxel_size)
                         for voxel in bucket[0].keys()])

                    if voxels.shape == (0,):
                        continue
                    bbox_min = np.min(np.vstack(
                        ([bbox_min] if bbox_min is not None else [])
                        + [voxels]), axis=0)
                    bbox_max = np.max(np.vstack(
                        ([bbox_max] if bbox_max is not None else [])
                        + [voxels]), axis=0)

        print('bounding box min:', bbox_min)
        print('bounding box max:', bbox_max)

        return bbox_min, bbox_max

    def get_bounding_boxes(self, subjects):
        """get bounding boxes of the chosen sulcus for all subjects

      Function that outputs the bounding box for the listed sulci on a manually
      labeled dataset.
      Bounding box corresponds to the biggest box encountered in the manually
      labeled subjects in the Talairach space. The bounding box is the smallest
      rectangular box that encompasses the sulcus.

      Parameters:
        subjects: list containing all subjects to be analyzed

      Returns:
        list_bbmin: list containing the upper right vertex of the box
                    in the Talairach space
        list_bbmax: list containing the lower left vertex of the box
                    in the Talairach space
      """

        # Initialization
        list_bbmin = []
        list_bbmax = []

        for sub in subjects:
            print(sub)

            sulci_pattern = sub['dir'] \
                            + '%(subject)s/t1mri/t1/default_analysis/folds/3.3/' \
                              'base2018_manual/' \
                              '%(side)s%(subject)s_base2018_manual.arg'

            bbox_min, bbox_max = self.get_one_bounding_box(sulci_pattern % sub)

            list_bbmin.append([bbox_min[0], bbox_min[1], bbox_min[2]])
            list_bbmax.append([bbox_max[0], bbox_max[1], bbox_max[2]])

        return list_bbmin, list_bbmax

    @staticmethod
    def compute_box_talairach_space(list_bbmin, list_bbmax):
        """Returns the coordinates of the box in Talairach space

      Parameters:
        list_bbmin: list containing the upper right vertex of the box
                    in the Talairach space
        list_bbmax: list containing the lower left vertex of the box
                    in the Talairach space

      Returns:
        bbmin_tal: numpy array with the x,y,z coordinates
                    of the upper right corner of the box
        bblax_tal: numpy array with the x,y,z coordinates
                    of the lower left corner of the box
      """

        bbmin_tal = np.array(
            [min([val[0] for k, val in enumerate(list_bbmin)]),
             min([val[1] for k, val in enumerate(list_bbmin)]),
             min([val[2] for k, val in enumerate(list_bbmin)])])

        bbmax_tal = np.array(
            [max([val[0] for k, val in enumerate(list_bbmax)]),
             max([val[1] for k, val in enumerate(list_bbmax)]),
             max([val[2] for k, val in enumerate(list_bbmax)])])

        return bbmin_tal, bbmax_tal

    @staticmethod
    def transform_tal_to_native():
        """Returns the transformation from Talairach space to MNI template

      Compute the transformation from Talairach space to MNI space, passing through SPM template.
      Empirically, this was done because some Deep learning results were better with SPM template.
      The transform from MNI to SPM template is taken from HCP database

      Parameters:

      Returns:
        tal_to_native: transformation used from Talairach space to native MRI space
        voxel_size: voxel size (in MNI referential or HCP normalized SPM space)
      """

        # Gets the transformation file from brainvisa directory structure
        tal_to_spm_template = aims.read(
            aims.carto.Paths.findResourceFile(
                'transformation/talairach_TO_spm_template_novoxels.trm'))

        # Gets a normalized SPM file from the morphologist analysis of the hcp database
        image_normalized_spm = aims.read(_image_normalized_SPM)

        # Tranformation from the native space to the MNI/SPM template referential
        native_to_spm_template = aims.AffineTransformation3d(
            image_normalized_spm.header()['transformations'][-1])

        # Tranformation from the Talairach space to the native space
        tal_to_native = native_to_spm_template.inverse() * tal_to_spm_template

        voxel_size = image_normalized_spm.header()['voxel_size'][:3]

        return tal_to_native, voxel_size

    @staticmethod
    def compute_box_voxel(bbmin_tal, bbmax_tal, tal_to_native, voxel_size):
        """Returns the coordinates of the box as voxels encompassing the sulcus for all subjects

      Coordinates of the box in voxels are determined in the MNI referential

      Parameters:
        bbmin_tal: numpy array with the coordinates of the upper right corner of the box (Talairach space)
        bbmax_tal: numpy array with the coordinates of the lower left corner of the box (Talairach space)
        tal_to_native: transformation used from Talairach space to native MRI space
        voxel_size: voxel size (in MNI referential or HCP normalized SPM space)

      Returns:
        bbmin_vox: numpy array with the coordinates of the upper right corner of the box (voxels in MNI space)
        bblax_vox: numpy array with the coordinates of the lower left corner of the box (voxels in MNI space)
      """

        # Application of the transformation to bbox
        bbmin_mni = tal_to_native.transform(bbmin_tal)
        bbmax_mni = tal_to_native.transform(bbmax_tal)

        # To go back from mms to voxels
        bbmin_vox = np.round(np.array(bbmin_mni) / voxel_size).astype(int)
        bbmax_vox = np.round(np.array(bbmax_mni) / voxel_size).astype(int)

        return bbmin_vox, bbmax_vox

    def compute_bounding_box(self, number_subjects=_ALL_SUBJECTS):
        """Main class program to compute the bounding box
        """

        if number_subjects:
            subjects = self.list_all_subjects()

            # Gives the possibility to list only the first number_subjects
            subjects = (
                subjects
                if number_subjects == _ALL_SUBJECTS
                else subjects[:number_subjects])

            # Determine the box encompassing the sulcus for all subjects
            # The coordinates are determined in Talairach space
            list_bbmin, list_bbmax = self.get_bounding_boxes(subjects)
            bbmin_tal, bbmax_tal = self.compute_box_talairach_space(list_bbmin,
                                                                    list_bbmax)

            # Compute the transform from the Talairach space to native MRI space
            tal_to_native, voxel_size = self.transform_tal_to_native()

            # Determine the box encompassing the _sulcus for all subjects
            # The coordinates are determined in voxels in MNI space
            bbmin_vox, bbmax_vox = self.compute_box_voxel(bbmin_tal,
                                                          bbmax_tal,
                                                          tal_to_native,
                                                          voxel_size)

            print("box: min = ", bbmin_vox)
            print("box: max = ", bbmax_vox)

        return bbmin_vox, bbmax_vox


def max_bounding_box(src_dir=_SRC_DIR_DEFAULT, sulcus=_SULCUS_DEFAULT,
                     number_subjects=_ALL_SUBJECTS):
    """ Main program computing the box encompassing the sulcus in all subjects

  The programm loops over all subjects
  and computes in MNI space the voxel coordinates of the box encompassing
  the sulci for all subjects

  Args:
      src_dir: list of strings -> directories of the supervised databases
      sulcus: list of strings giving the sulci to analyze
  """

    box = BoundingBoxMax(src_dir=src_dir, sulcus=sulcus)
    # List all subjects from _root_dir
    bbmin_vox, bbmax_vox = box.compute_bounding_box(
        number_subjects=number_subjects)

    return bbmin_vox, bbmax_vox


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        src_dir: a list with source directory names, full path
        sulcus: a list containing the sulci to analyze
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='bbox_definition.py',
        description='Computes bounding box around the named sulci')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box. '
             '0 subject is allowed, for debug purpose.'
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'.'
             '0 subject is allowed, for debug purpose.'
             'Default is : all')

    args = parser.parse_args(argv)
    src_dir = args.src_dir  # src_dir is a list
    sulcus = args.sulcus  # sulcus is a list

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
            "nb_subjects must be either the string \"all\" or an integer")

    return src_dir, sulcus, number_subjects


def main(argv):
    """Reads argument line and determines the max bounding box

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        src_dir, sulcus, number_subjects = parse_args(argv)
        # Actual API
        max_bounding_box(src_dir, sulcus, number_subjects)
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
