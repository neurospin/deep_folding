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
in the normalized SPM space
"""

from __future__ import division
from __future__ import print_function

import sys
import glob
import os
from os.path import join
import argparse
import six

import numpy as np

from soma import aims
from deep_folding.anatomist_tools.utils.logs import LogJson
from deep_folding.anatomist_tools.utils.bbox import compute_max_box
from deep_folding.anatomist_tools.utils.sulcus_side import complete_sulci_name

_ALL_SUBJECTS = -1

# Default directory in which lies the manually segmented database
_SRC_DIR_DEFAULT = "/neurospin/lnao/PClean/database_learnclean/all/"

# Default directory to which we write the bounding box results
_TGT_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/bbox"

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'L'

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'S.T.s.ter.asc.ant.'

# Gives the relative path to the manually labelled graph .arg
# in the supervise
_PATH_TO_GRAPH_DEFAULT = "t1mri/t1/default_analysis/folds/3.3/base2018_manual"

# A normalized SPM image to get the voxel size
_IMAGE_NORMALIZED_SPM_DEFAULT = '/neurospin/hcp/' \
                        'ANALYSIS/3T_morphologist/100206/' \
                        't1mri/default_acquisition/normalized_SPM_100206.nii'


class BoundingBoxMax:
    """Determines the maximum Bounding Box around given sulci

    It is determined in the normalized SPM referential
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 image_normalized_spm=_IMAGE_NORMALIZED_SPM_DEFAULT,
                 out_voxel_size=None):
        """Inits with list of directories and list of sulci

        Args:
            src_dir: list of strings naming ful path source directories
            path_to_graph: list of strings naming relative path to labelle graph
            tgt_dir: name of target directory with full path
            sulcus: sulcus name
            side: hemisphere side (either L for left, or R for right hemisphere)
            image_normalized_spm: string giving file name (with path) of
                normalized SPM file out of which is extracted the voxel size
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
        self.tgt_dir = tgt_dir
        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.image_normalized_spm = image_normalized_spm
        self.out_voxel_size = out_voxel_size


        # Json full name is the name of the sulcus + .json
        # and is kept under the subdirectory Left or Right
        json_file = join(self.tgt_dir, self.side, self.sulcus + '.json')
        self.json = LogJson(json_file)

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
                                     'graph_file': graph_file}
                        subjects.append(subject_d)

        return subjects

    def get_one_bounding_box_aims_talairach(self, graph_filename):
        """get bounding box of the chosen sulcus for one data graph

      Function that outputs the bounding box for the listed sulci
      for this datagraph. The bounding box is the smallest rectangular box
      that encompasses the chosen sulcus.
      It is given in the AIMS Talairch referential, different from the MNI
      Talairach referential.

      Parameters:
        graph_filename: string being the name of graph file .arg to analyze:
                        for example: 'Lammon_base2018_manual.arg'

      Returns:
        bbox_min: numpy array giving the upper right vertex coordinates
                of the box in the Talairach space
        bbox_max: numpy array fiving the lower left vertex coordinates
                of the box in the Talairach space
      """

        # Reads the data graph and transforms it to AIMS Talairach referential
        # Note that this is NOT the MNI Talairach referential
        # This is the Talairach referential used in AIMS
        # There are several Talairach referentials
        graph = aims.read(graph_filename)
        voxel_size = graph['voxel_size'][:3]
        tal_transfo = aims.GraphManip.talairach(graph)
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

        print('box (AIMS Talairach) min:', bbox_min)
        print('box (AIMS Talairach) max:', bbox_max)

        return bbox_min, bbox_max

    def get_bounding_boxes(self, subjects):
        """get bounding boxes of the chosen sulcus for all subjects

      Function that outputs the bounding box for the listed sulci on a manually
      labeled dataset.
      Bounding box corresponds to the biggest box encountered in the manually
      labeled subjects in the AIMS Talairach space, different from the MNI
      Talairach template.
      The bounding box is the smallest rectangular box that
      encompasses the sulcus.

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
            # Its substitutes 'subject' in graph_file name
            graph_file = sub['graph_file'] % sub
            # It looks for a graph file .arg
            sulci_pattern = glob.glob(join(sub['dir'], graph_file))[0]

            bbox_min, bbox_max = self.get_one_bounding_box(sulci_pattern % sub)

            list_bbmin.append([bbox_min[0], bbox_min[1], bbox_min[2]])
            list_bbmax.append([bbox_max[0], bbox_max[1], bbox_max[2]])

        return list_bbmin, list_bbmax

    def tal_to_normalized_spm(self):
        """Returns the transformation from AIMS Talairach to normalized SPM

      Computes the transformation from AIMS Talairach space to normalized SPM
      space, MNI space, passing through SPM template.
      Empirically, this was done because some Deep learning results were better
      with SPM template.
      The transform from MNI to SPM template is taken from HCP database

      Returns:
        tal_to_normalized_spm: transformation from AIMS Talairach space to
                    normalized SPM
        voxel_size: voxel size (in MNI referential or HCP normalized SPM space)
      """

        # Gets the transformation file from brainvisa directory structure
        # Transforms from AIMS Talairach to the true MNI space with the origin
        # at the center
        # It is in /casa/install/share/brainvisa-share-5.0/transformation
        tal_to_spm_template = aims.read(
            aims.carto.Paths.findResourceFile(
                'transformation/talairach_TO_spm_template_novoxels.trm'))

        # Gets a normalized SPM file from the morphologist analysis
        image_normalized_spm = aims.read(self.image_normalized_spm)

        # Tranformation from the normalized SPM
        # to the template SPM
        # normalized_spm_to_spm_template = aims.AffineTransformation3d(
        #    image_normalized_spm.header()['transformations'][-1])
        normalized_spm_to_spm_template = aims.read(
            aims.carto.Paths.findResourceFile(
                'transformation/spm_template_TO_spm_template_novoxels.trm'))

        # Tranformation from the Talairach space to the native space
        tal_to_normalized_spm = normalized_spm_to_spm_template.inverse() \
                                * tal_to_spm_template
        if self.out_voxel_size:
            voxel_size = self.out_voxel_size
        else:
            voxel_size = image_normalized_spm.header()['voxel_size'][:3]

        return tal_to_normalized_spm, voxel_size

    @staticmethod
    def compute_box_voxel(bbmin_tal, bbmax_tal,
                          tal_to_normalized_spm, voxel_size):
        """Returns the coordinates of the box as voxels

      Coordinates of the box in voxels are determined in the MNI referential

      Parameters:
        bbmin_tal: numpy array with the coordinates of the upper right corner
                of the box (AIMS Talairach space)
        bbmax_tal: numpy array with the coordinates of the lower left corner
                of the box (AIMS Talairach space)
        tal_to_normalized_spm: transformation used from Talairach space
                to normalized SPM
        voxel_size: voxel size (in MNI referential or HCP normalized SPM space)

      Returns:
        bbmin_vox: numpy array with the coordinates of the upper right corner
                of the box (voxels in MNI space)compite_b
        bblax_vox: numpy array with the coordinates of the lower left corner
                of the box (voxels in MNI space)
      """

        # Application of the transformation to bbox
        bbmin_mni = tal_to_normalized_spm.transform(bbmin_tal)
        bbmax_mni = tal_to_normalized_spm.transform(bbmax_tal)

        # To go back from mms to voxels
        bbmin_vox = np.round(np.array(bbmin_mni) / voxel_size).astype(int)
        bbmax_vox = np.round(np.array(bbmax_mni) / voxel_size).astype(int)

        return bbmin_vox, bbmax_vox

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

            # Creates target dir if it doesn't exist
            if not os.path.exists(self.tgt_dir):
                os.mkdir(self.tgt_dir)

            # Writes number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(subjects),
                           'src_dir': self.src_dir,
                           'tgt_dir': self.tgt_dir,
                           'out_voxel_size': 1 if self.out_voxel_size is None else self.out_voxel_size}
            self.json.update(dict_to_add=dict_to_add)

            # Determines the box encompassing the sulcus for all subjects
            # The coordinates are determined in AIMS Talairach space
            list_bbmin, list_bbmax = self.get_bounding_boxes(subjects)
            bbmin_tal, bbmax_tal = compute_max_box(list_bbmin, list_bbmax)

            dict_to_add = {'bbmin_AIMS_Talairach': bbmin_tal.tolist(),
                           'bbmax_AIMS_Talairach': bbmax_tal.tolist()}

            # Computes the transform from the AIMS Talairach space
            # to normalized SPM space
            tal_to_normalized_spm, voxel_size = self.tal_to_normalized_spm()

            # Determines the box encompassing the sulcus for all subjects
            # The coordinates are determined in voxels in MNI space
            bbmin_vox, bbmax_vox = self.compute_box_voxel(bbmin_tal,
                                                          bbmax_tal,
                                                          tal_to_normalized_spm,
                                                          voxel_size)

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


def bounding_box(src_dir=_SRC_DIR_DEFAULT, tgt_dir=_TGT_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                 sulcus=_SULCUS_DEFAULT, side=_SIDE_DEFAULT,
                 number_subjects=_ALL_SUBJECTS,
                 image_normalized_spm=_IMAGE_NORMALIZED_SPM_DEFAULT,
                 out_voxel_size=None):
    """ Main program computing the box encompassing the sulcus in all subjects

  The programm loops over all subjects
  and computes in MNI space the voxel coordinates of the box encompassing
  the sulci for all subjects

  Args:
      src_dir: list of strings -> directories of the supervised databases
      tgt_dir: string giving target directory path
      path_to_graph: string giving relative path to manually labelled graph
      side: hemisphere side (either 'L' for left, or 'R' for right)
      sulcus: string giving the sulcus to analyze
      number_subjects: integer giving the number of subjects to analyze,
            by default it is set to _ALL_SUBJECTS (-1).
      image_normalized_spm: string giving file name (with path) of normalized
            SPM file out of which is extracted the voxel size
  """

    box = BoundingBoxMax(src_dir=src_dir, tgt_dir=tgt_dir,
                         path_to_graph=path_to_graph,
                         sulcus=sulcus, side=side,
                         image_normalized_spm=image_normalized_spm,
                         out_voxel_size=out_voxel_size)
    bbmin_vox, bbmax_vox = box.compute_bounding_box(
        number_subjects=number_subjects)

    return bbmin_vox, bbmax_vox


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: a dictionary with all arugments as keys
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='bounding_box.py',
        description='Computes bounding box around the named sulcus')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT, nargs='+',
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the output bbox json files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box. '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-m", "--image_normalized_SPM", type=str,
        default=_IMAGE_NORMALIZED_SPM_DEFAULT,
        help='Name (with path) of normalized SPM image. '
             'It is used to determine voxel size. '
             'Default is : ' + _IMAGE_NORMALIZED_SPM_DEFAULT)
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
        "-v", "--out_voxel_size", type=int, default=None,
        help='Voxel size of of bounding box. '
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)
    params['src_dir'] = args.src_dir  # src_dir is a list
    params['path_to_graph'] = args.path_to_graph
    params['tgt_dir']= args.tgt_dir # tgt_dir is a string, only one directory
    params['image_normalized_spm'] = args.image_normalized_SPM
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
                     tgt_dir=params['tgt_dir'],
                     sulcus=params['sulcus'], side=params['side'],
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
