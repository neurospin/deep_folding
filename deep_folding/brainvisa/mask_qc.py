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

"""Quality check of generated masks

"""

import argparse
import os
import random
import sys

import numpy as np
import scipy.ndimage
from soma import aims
from soma import aimsalgo as ago

_SULCUS_DEFAULT = 'S.C.'
_SIDE_DEFAULT = 'R'
_OUT_VOXEL_SIZE = (2, 2, 2)
_INPUT_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/new_v1/mask"
_OUTPUT_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/new_v1/QC_skeleton"


def check(
        side=_SIDE_DEFAULT,
        sulcus=_SULCUS_DEFAULT,
        out_voxel_size=_OUT_VOXEL_SIZE,
        nb_subjects=1):
    """
    """
    morpho_dir = "/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist"
    vs = out_voxel_size[0]
    side = 'right' if side == 'R' else 'left'

    list_sub = os.listdir(morpho_dir)
    subject_list = random.sample(list_sub, nb_subjects)
    subject_list = ['146533', '334635', '304727', '665254', '585256',
                    '303119', '299760', '877269', '194140', '552544']

    for subject in subject_list:
        # Loading of subject graph
        graph_dir = f"{morpho_dir}/{subject}/" + \
                    "t1mri/default_acquisition/default_analysis/folds/3.1/" + \
                    f"default_session_auto/R{subject}_default_session_auto.arg"
        graph = aims.read(graph_dir)

        # Loading of subject skeleton
        skeleton_dir = f"{morpho_dir}/{subject}/" +\
                       "t1mri/default_acquisition/default_analysis/" + \
                       f"segmentation/Rskeleton_{subject}.nii.gz"
        skeleton = aims.read(skeleton_dir)

        masked_resampled = aims.Volume(
            skeleton.header()['volume_dimension'][:3],
            dtype=skeleton.__array__().dtype)
        masked_resampled.header()['voxel_size'] = skeleton.header()[
            'voxel_size'][:3]

        g_to_icbm = aims.GraphManip.getICBM2009cTemplateTransform(graph)

        g_to_rw = g_to_icbm.inverse()

        mask = aims.read(
            f"{_INPUT_DIR_DEFAULT}/{vs}mm/R/{sulcus}_{side}.nii.gz")

        resampler = ago.ResamplerFactory(mask).getResampler(0)
        resampler.setDefaultValue(0)
        resampler.setRef(mask)
        resampler.resample(mask, g_to_rw, 0, masked_resampled)

        arr = np.asarray(masked_resampled)
        arr_filter = scipy.ndimage.gaussian_filter(
            arr.astype(float),
            sigma=0.5,
            order=0,
            output=None,
            mode='reflect',
            truncate=4.0)
        arr[:] = (arr_filter > 0.001).astype(int)

        masked_resampled_f = aims.Volume(arr)
        masked_resampled_f.header()['voxel_size'] = skeleton.header()[
            'voxel_size'][:3]

        aims.write(
            masked_resampled_f,
            f"{_OUTPUT_DIR_DEFAULT}/{vs}mm/"
            f"mask_resampled_{subject}_{sulcus}_{side}.nii.gz")


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='mask_qc.py',
        description='Generates masks in subject skeleton space')

    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box. '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-v", "--out_voxel_size", type=int, nargs='+', default=_OUT_VOXEL_SIZE,
        help='Voxel size of output images'
             'Default is : 1 1 1')
    parser.add_argument(
        "-n", "--nb_subjects", type=int, default=None,
        help='Number of subjects to take into account.'
             'If not precised, all subjects are processed'
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)
    params['sulcus'] = args.sulcus  # a list of sulci
    params['side'] = args.side
    params['out_voxel_size'] = tuple(args.out_voxel_size)
    params['nb_subjects'] = args.nb_subjects

    return params


def main(argv):
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    check(side=params['side'],
          sulcus=params['sulcus'],
          out_voxel_size=params['out_voxel_size'],
          nb_subjects=params['nb_subjects'])

######################################################################
# Main program
######################################################################


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
