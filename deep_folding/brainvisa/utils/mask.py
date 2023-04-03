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

import json
from os.path import join

import deep_folding.brainvisa.utils.dilate_mask as dl
import numpy as np
from scipy import ndimage
from soma import aims
from soma.aimsalgo import MorphoGreyLevel_S16

from deep_folding.config.logs import set_file_logger

from deep_folding.brainvisa.utils.constants import \
    _MASK_DIR_DEFAULT, _DILATION_DEFAULT, _THRESHOLD_DEFAULT

# Defines logger
log = set_file_logger(__file__)


_AIMS_BINARY_ONE = 32767


def compute_bbox_mask(arr):

    # Gets location of bounding box as slices
    objects_in_image = ndimage.find_objects(arr)
    print(f"ndimage.find_objects(arr) = {objects_in_image}")
    if not objects_in_image:
        raise ValueError("There are only 0s in array!!!")

    loc = objects_in_image[0]
    bbmin = []
    bbmax = []

    for slicing in loc:
        bbmin.append(slicing.start)
        bbmax.append(slicing.stop)

    return np.array(bbmin), np.array(bbmax)


def compute_simple_mask(
        sulci_list,
        side,
        mask_dir=_MASK_DIR_DEFAULT,
        dilation=_DILATION_DEFAULT,
        threshold=_THRESHOLD_DEFAULT):
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
        log.info(f"mask file: {mask_file}")
        list_masks.append(aims.read(mask_file))

    # Computes the mask being a combination of all masks
    mask_result = list_masks[0]
    if len(list_masks) == 1:
        log.info(f"only one sulcus: {sulci_list[0]}")
        mask_result[np.asarray(mask_result) <= threshold] = 0
        arr_result = np.asarray(dl.dilate(mask_result, radius=dilation))
        np.asarray(mask_result)[:] = arr_result

    else:
        arr_result = np.asarray(mask_result)
        log.info(f"first sulcus: {sulci_list[0]}")
        for k, mask in enumerate(list_masks[1:]):
            print(f"sulcus {sulci_list[k+1]}")
            arr = np.asarray(mask)
            arr_result += arr

        mask_result[np.asarray(mask_result) <= threshold] = 0
        arr_result = np.asarray(dl.dilate(mask_result, radius=dilation))
        np.asarray(mask_result)[:] = arr_result

    log.info(f"threshold = {threshold}")
    # Computes the mask bounding box
    bbmin, bbmax = compute_bbox_mask(arr_result)
    # aims.write(mask_result, '/tmp/test.nii.gz')
    return mask_result, bbmin, bbmax


def intersect_binary(a, b):
    """returns intersection of two binary arrays"""
    return ((a + b) > 1).astype(np.int16)


def compute_intersection_mask(
        sulci_list,
        side,
        mask_dir=_MASK_DIR_DEFAULT,
        dilation=_DILATION_DEFAULT,
        threshold=_THRESHOLD_DEFAULT):
    """Function returning mask making intersection of masks over several sulci

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
    if len(list_masks) == 1:
        print(f"only one sulcus: {sulci_list[0]}")
        mask_result[np.asarray(mask_result) <= threshold] = 0
        arr_result = np.asarray(dl.dilate(mask_result, radius=dilation))
        np.asarray(mask_result)[:] = arr_result

    else:
        arr_result = np.asarray(mask_result)
        arr_result = (arr_result > 0).astype(np.int16)
        print(f"first sulcus: {sulci_list[0]}")
        for k, mask in enumerate(list_masks[1:]):
            print(f"sulcus {sulci_list[k+1]}")
            arr = np.asarray(mask)
            arr = (arr > 0).astype(np.int16)
            arr_result = intersect_binary(arr_result, arr)

        np.asarray(mask_result)[:] = arr_result
        mask_result[np.asarray(mask_result) <= threshold] = 0
        arr_result = np.asarray(dl.dilate(mask_result, radius=dilation))
        np.asarray(mask_result)[:] = arr_result
        print(
            f"np.unique(mask_result) = "
            f"{np.unique(np.asarray(mask_result), return_counts=True)}")

    print(
        f"np.unique after intersection = "
        f"{np.unique(arr_result, return_counts=True)}")
    print(f"Shape after intersection = {arr_result.shape}")

    # Computes the mask bounding box
    bbmin, bbmax = compute_bbox_mask(arr_result)
    return mask_result, bbmin, bbmax


# SPECIFIC FOR THE CINGULATE REGION STUDY (2022, CHAVAS, GAUDIN)
def compute_centered_mask(sulci_list, side, mask_dir=_MASK_DIR_DEFAULT):
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

    # Initializes and fills list of 2 masks, each represented as an aims volume
    list_masks = []
    hdr = aims.StandardReferentials.icbm2009cTemplateHeader()

    for sulcus in sulci_list:
        mask_file = join(mask_dir, side, sulcus + '.nii.gz')
        list_masks.append(aims.read(mask_file))

    # Threshold and dilation of first mask
    eligible_mask_1 = np.asarray(list_masks[0])
    eligible_mask_1[eligible_mask_1 < 10] = 0
    eligible_mask_1 = dl.dilate(list_masks[0])
    aims.write(eligible_mask_1, '/tmp/eligible_mask_1.nii.gz')

    # Threshold of other mask
    eligible_mask_2 = np.asarray(list_masks[1])
    eligible_mask_2[eligible_mask_2 < 10] = 0
    eligible_mask_2[eligible_mask_2 >= 10] = 1
    aims.write(list_masks[1], '/tmp/eligible_mask_2.nii.gz')

    # Intersection of the two eligible masks
    intersec_mask = aims.Volume(list_masks[0].shape, dtype='S16')
    intersec_mask.copyHeaderFrom(hdr)
    intersec_mask.header()['voxel_size'] = [2, 2, 2]
    intersec_mask_arr = np.asarray(intersec_mask)
    intersec_mask_arr[:] = eligible_mask_1 & eligible_mask_2
    aims.write(intersec_mask, '/tmp/intersec_mask.nii.gz')

    # Dilation of intersec_mask
    morpho = MorphoGreyLevel_S16()
    intersec_mask_arr[intersec_mask_arr >= 1] = _AIMS_BINARY_ONE
    intersec_mask = morpho.doDilation(intersec_mask, 15.0)
    intersec_mask_arr = np.asarray(intersec_mask)
    intersec_mask_arr[intersec_mask_arr >= 1] = 1
    aims.write(intersec_mask, '/tmp/intersec_mask_dilated.nii.gz')

    # Intersection of intersec_mask, eligible_mask_1 and eligible_mask_2
    mask_result = aims.Volume(list_masks[0].shape, dtype='S16')
    mask_result.copyHeaderFrom(hdr)
    mask_result.header()['voxel_size'] = [2, 2, 2]
    mask_result_arr = np.asarray(mask_result)

    intersec_mask_arr = np.asarray(intersec_mask)
    intersec_1 = intersec_mask_arr.copy() & np.asarray(eligible_mask_1)
    intersec_2 = intersec_mask_arr & np.asarray(eligible_mask_2)

    mask_result_arr[:] = intersec_1 + intersec_2
    mask_result_arr[mask_result_arr > 1] = 1

    aims.write(mask_result, '/tmp/mask_result.nii.gz')

    # Computes the mask bounding box
    bbmin, bbmax = compute_bbox_mask(mask_result_arr)

    return mask_result, bbmin, bbmax


if __name__ == '__main__':
    arr_mask, bbmin, bbmax = compute_centered_mask(['paracingular._right',
                                                    'F.C.M.ant._right'],
                                                   'R')
    print("bbmin = ", bbmin)
    print("bbmax = ", bbmax)
