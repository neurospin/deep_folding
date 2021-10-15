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

""" Getting bounding box from a list of sulci and specific hemisphere

The aim of this script is to output the bounding box of one or several sulci of
a specified hemisphere.

"""

from os.path import join
from soma import aims
from scipy import ndimage
import numpy as np
import json


_MASK_DIR_DEFAULT = "/neurospin/dico/deep_folding_data/data/mask/"


def compute_bbox_mask(arr):
    
    # Gets location of bounding box as slices
    loc = ndimage.find_objects(arr)[0]
    bbmin = []
    bbmax = []
    
    for slicing in loc:
        bbmin.append(slicing.start)
        bbmax.append(slicing.stop)

    return bbmin, bbmax

def compute_mask(sulci_list, side, mask_dir=_MASK_DIR_DEFAULT):
    """Function returning mask combining mask over several sulci

    It reads mask files in the source mask directory and combines them.
    They are listed in subdirectory 'L' or 'R' according the hemisphere

    Args:
        sulci_list: a list of sulci
        side: a string corresponding to the hemisphere, whether 'L' or 'R'
        mask_dir: path to source directory containing masks

    Returns:
        mask_result: AIMS volume containing combined mask
        bbmin: an array of minimum coordinates of min box around the mask
        bbmax: an array of maximum coordinates of max_box around the mask
    """

    # Initializes and fills list of masks, each repreented as an aims volume
    list_masks = []
    
    for sulcus in sulci_list:
        mask_file = join(mask_dir, side, sulcus + '.nii.gz')
        list_masks.append(aims.read(mask_file))
    
    # Computes the mask being a combination of all masks
    mask_result = list_masks[0]
    
    arr_result = np.asarray(mask_result).astype(bool)
    for mask in list_masks[1:]:
        arr = np.asarray(mask)
        arr_result += arr.astype(bool)
        
    arr_result = arr_result.astype(int)
    np.asarray(mask_result)[:] = arr_result
    
    # Computes the mask bounding box
    bbmin, bbmax = compute_bbox_mask(arr_result)
        
    return mask_result, bbmin, bbmax



if __name__ == '__main__':
    arr_mask, bbmin, bbmax = compute_mask(['S.T.s.ter.asc.ant._left',
                                    'S.T.s.ter.asc.test._left'],
                                    'L')
    print("bbmin = ", bbmin)
    print("bbmax = ", bbmax)
