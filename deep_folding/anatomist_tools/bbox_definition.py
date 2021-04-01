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
Script that outputs bounding box for a given sulcus based on a manually
labeled dataset.
Bounding box corresponds to the biggest box that encompasses the given sulcus 
got all subjects of the manually labelled dataset.
"""

######################################################################
# Imports 
######################################################################

from soma import aims
import os
import numpy as np

######################################################################
# Global variables (that the user can change)
######################################################################

# _root_dir is the directory in which lies the manually segmented database
_root_dir = "/neurospin/lnao/PClean/database_learnclean/all/"

# hemisphere 'L' or 'R'
_side = 'L'

# sulcus to encompass:
# its name depends on the hemisphere side
_sulcus = ('S.T.s.ter.asc.ant._left' if _side == 'L'
           else 'S.T.s.ter.asc.ant._right')

######################################################################
# Global variables (that the user will probably not change)
######################################################################

# A normalized SPM image to get the HCP morphologist transformation
_image_normalized_SPM = '/neurospin/hcp/ANALYSIS/3T_morphologist/100206/t1mri/default_acquisition/normalized_SPM_100206.nii'


######################################################################
# Functions
######################################################################

def list_all_subjects():
    """List all subjects from the clean database (directory _root_dir).

    Subjects are the names of the subdirectories of the root directory.

    Parameters:

    Returns:
        subjects: a list containing all subjects to be analyzed
    """

    subjects = []

    # Main loop: list all subjects of the directory _root_dir
    for filename in os.listdir(_root_dir):
        directory = os.path.join(_root_dir, filename)
        if os.path.isdir(directory):
            if filename != 'ra':
                subjects.append(filename)

    return subjects


def get_one_bounding_box(graph_filename):
    """get bounding box of the chosen sulcus for one data graph

  Function that outputs the bounding box for the sulcus '_sulcus' for this datagraph.
  The bounding box is the smallest
  rectangular box that encompasses the chosen sulcus

  Parameters:
    graph_filename: string being the name of graph file .arg to be analyzed: 'Lammon_base2018_manual.arg'

  Returns:
    bbox_min: numpy array giving the upper right vertex coordinates of the box in the Talairach space
    bbox_max: numpy array fiving the lower left vertex coordinates of the box in the Talairach space
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
        if vname != _sulcus:
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


def get_bounding_boxes(subjects):
    """get bounding boxes of the chosen sulcus for all subjects

  Function that outputs the bounding box for the sulcus '_sulcus' on a manually
  labeled dataset.
  Bounding box corresponds to the biggest box encountered in the manually
  labeled subjects in the Talairach space. The bounding box is the smallest
  rectangular box that encompasses the while sulcus

  Parameters:
    subjects: list containing all subjects to be analyzed

  Returns:
    list_bbmin: list containing the upper right vertex of the box in the Talairach space
    list_bbmax: list containing the lower left vertex of the box in the Talairach space
  """

    # Initialization
    atts = {'subject': '', 'side': _side}
    list_bbmin = []
    list_bbmax = []
    sulci_pattern = _root_dir \
                    + '%(subject)s/t1mri/t1/default_analysis/folds/3.3/base2018_manual/%(side)s%(subject)s_base2018_manual.arg'

    for sub in subjects:
        print(sub)

        atts['subject'] = sub

        bbox_min, bbox_max = get_one_bounding_box(sulci_pattern % atts)

        list_bbmin.append([bbox_min[0], bbox_min[1], bbox_min[2]])
        list_bbmax.append([bbox_max[0], bbox_max[1], bbox_max[2]])

    return list_bbmin, list_bbmax


def compute_box_talairach_space(list_bbmin, list_bbmax):
    """Returns the coordinates of the box in Talairach space encompassing the sulcus for all subjects

  Parameters:
    list_bbmin: list containing the upper right vertex of the box in the Talairach space
    list_bbmax: list containing the lower left vertex of the box in the Talairach space    

  Returns:
    bbmin_tal: numpy array with the x,y,z coordinates of the upper right corner of the box
    bblax_tal: numpy array with the x,y,z coordinates of the lower left corner of the box
  """

    bbmin_tal = np.array([min([val[0] for k, val in enumerate(list_bbmin)]),
                          min([val[1] for k, val in enumerate(list_bbmin)]),
                          min([val[2] for k, val in enumerate(list_bbmin)])])

    bbmax_tal = np.array([max([val[0] for k, val in enumerate(list_bbmax)]),
                          max([val[1] for k, val in enumerate(list_bbmax)]),
                          max([val[2] for k, val in enumerate(list_bbmax)])])

    return bbmin_tal, bbmax_tal


def compute_transform_tal_to_native():
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
        aims.carto.Paths.findResourceFile('transformation/talairach_TO_spm_template_novoxels.trm'))

    # Gets a normalized SPM file from the morphologist analysis of the hcp database
    image_normalized_spm = aims.read(_image_normalized_SPM)

    # Tranformation from the native space to the MNI/SPM template referential
    native_to_spm_template = aims.AffineTransformation3d(image_normalized_spm.header()['transformations'][-1])

    # Tranformation from the Talairach space to the native space
    tal_to_native = native_to_spm_template.inverse() * tal_to_spm_template

    voxel_size = image_normalized_spm.header()['voxel_size'][:3]

    return tal_to_native, voxel_size


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


def main():
    """ Main program determining the box encompassing the _sulcus in all subjects

  The programm loops over all subjects
  and computes in MNI space the voxel coordinates of the box encompassing
  the sulcus '_sulcus' for all subjects  
  """

    # List all subjects from _root_dir
    subjects = list_all_subjects()

    # Determine the box encompassing the _sulcus for all subjects
    # The coordinates are determined in Talairach space
    list_bbmin, list_bbmax = get_bounding_boxes(subjects)
    bbmin_tal, bbmax_tal = compute_box_talairach_space(list_bbmin, list_bbmax)

    # Compute the transform from the Talairach space to native MRI space
    tal_to_native, voxel_size = compute_transform_tal_to_native()

    # Determine the box encompassing the _sulcus for all subjects
    # The coordinates are determined in voxels in MNI space
    bbmin_vox, bbmax_vox = compute_box_voxel(bbmin_tal, bbmax_tal, tal_to_native, voxel_size)

    print("box: min = ", bbmin_vox)
    print("box: max = ", bbmax_vox)

    return bbmin_vox, bbmax_vox


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    main()
