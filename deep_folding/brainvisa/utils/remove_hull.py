#!/usr/bin/env python
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

""" Removing hull voxels from skeletons' crops

The aim of this script is to remove skeleton's voxels that correspond to the
hull and to use this new volume for visualization if needed.

"""
import argparse
import glob
# system imports
import os
import sys

import dico_toolbox as dtx
import numpy as np
import pandas as pd
import six
from joblib import cpu_count
from pqdm.processes import pqdm
# pyAims import
from soma import aims

_AIMS_BINARY_ONE = 32767
_EXTERNAL = 11  # Value of external part
_INTERNAL = 0  # Value of the internal part of the brain

_DEFAULT_PADDING = 1  # local window in which to look for external valye
_DEFAULT_THRESHOLD = 12  # Looks for skeleton values >= _DEFAULT_THRESHOLD

_ALL_SUBJECTS = -1

# Input directory contaning the morphologist analysis of the HCP database
_SRC_DIR_DEFAULT = \
    '/nfs/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm/'

# Directory containing subjects meshes without hull once created
# default corresponds to
# -------------------------
_TGT_DIR_DEFAULT = \
    '/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm/meshes/'

_SIDE_DEFAULT = 'R'


def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus - 2, 1)


def remove_hull(arr, padding=_DEFAULT_PADDING, ext=_DEFAULT_PADDING):
    """Removes the pixels on the hull.

    Pixels on the hull are defined as being in contact
    with both the internal and external part.
    This function removes the hull in place

    Args:
        arr: numpy array modified in the array
        padding: padding of the image, equal to the extension ext
        ext: local array extension in which to look for extern/internal pixels
    """
    arr_pad = np.pad(arr,
                     ((padding, padding),
                      (padding, padding),
                      (padding, padding),
                      (0, 0)),
                     'constant',
                     constant_values=0)

    coords = np.argwhere(arr_pad > _EXTERNAL)

    for i, j, k, l in coords:
        if arr_pad[i, j, k, l] != _EXTERNAL \
           and arr_pad[i, j, k, l] != _INTERNAL:
            local_array = arr_pad[i - ext:i + ext + 1,
                                  j - ext:j + ext + 1, k - ext:k + ext + 1, l]
            if np.any(
                    local_array == _INTERNAL) and np.any(
                    local_array == _EXTERNAL):
                arr[i - padding, j - padding, k - padding, l] = 0


def threshold_and_binarize(arr, threshold=_DEFAULT_THRESHOLD):
    """Threshold images

    Args:
        arr: numpy array
    """
    arr[np.where(arr < threshold)] = 0
    arr[np.where(arr >= threshold)] = _AIMS_BINARY_ONE


def convert_volume_to_bucket(vol):
    """Converts volume to bucket

    Args:
        arr: numpy array
    """
    c = aims.Converter_rc_ptr_Volume_S16_BucketMap_VOID()
    bucket_map = c(vol)
    bucket = bucket_map[0]
    bucket = np.array([bucket.keys()[k].list()
                      for k in range(len(bucket.keys()))])
    return bucket_map, bucket


def create_one_mesh(
        vol,
        padding=_DEFAULT_PADDING,
        ext=_DEFAULT_PADDING,
        threshold=_DEFAULT_THRESHOLD):
    """Creates

    Args:
        vol: aims volume
    """

    arr = np.asarray(vol)

    # Removes hull
    remove_hull(arr, padding, ext)

    # Thresholds and "binarizes"
    threshold_and_binarize(arr, threshold)

    # Conversion of volume to bucket
    bucket_map, bucket = convert_volume_to_bucket(vol)

    # Conversion of bucket to mesh
    mesh = dtx._aims_tools.bucket_to_mesh(bucket_map[0])

    return bucket_map, bucket, mesh


class DatasetHullRemoved:
    """Generates meshes of crops without hull
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 side=_SIDE_DEFAULT,
                 number_subjects=_ALL_SUBJECTS,
                 list_subjects=None,
                 file_subjects=None):
        """Inits with list of directories and side

        Args:
            src_dir: string naming full path source directory,
                    containing crops images
            tgt_dir: name of target (output) directory with full path
            side: hemisphere side (L for left, or R for right hemisphere)
        """
        self.side = side
        self.src_dir = os.path.join(src_dir, f"{self.side}crops")
        self.tgt_dir = tgt_dir

        if list_subjects:
            self.list_subjects = list_subjects

        elif file_subjects:
            self.list_subjects = pd.read_csv(file_subjects)

        else:
            if number_subjects:
                # subjects are detected as the directory names under src_dir
                list_all_subjects = [
                    dI[:6] for dI in os.listdir(self.src_dir)
                    if ((os.path.isdir(self.src_dir)) and ('minf' not in dI))
                    ]

                # Gives the possibility to list only the first number_subjects
                self.list_subjects = (
                    list_all_subjects
                    if number_subjects == _ALL_SUBJECTS
                    else list_all_subjects[:number_subjects])

    def create_one_mesh(self, subject_id):
        """Creates one mesh from a skeleton crop (.nii file)

        Args:
            subject_id: string giving the subject ID
        """

        # Constant definition
        padding = _DEFAULT_PADDING
        ext = _DEFAULT_PADDING  # extent (equal to padding)
        threshold = _DEFAULT_THRESHOLD

        # Names directory where subject analysis files are stored
        subject_file = \
            os.path.join(self.src_dir, f"{subject_id}_normalized.nii.gz")

        # Reads nifti as AIMS volume and conversion to array
        vol = aims.read(subject_file)
        self.arr = np.asarray(vol)

        # Removes hull
        remove_hull(self.arr, padding, ext)

        # Thresholds and "binarizes"
        threshold_and_binarize(self.arr, threshold)

        # Conversion of volume to bucket
        bucket_map, bucket = convert_volume_to_bucket(vol)

        # Conversion of bucket to mesh
        m = dtx._aims_tools.bucket_to_mesh(bucket_map[0])

        # Writing of the mesh in tgt_dir folder
        aims.write(m, f"{self.tgt_dir}/mesh_{subject_id}.gii")
        return bucket

    def create_meshes(self):
        """Creates meshes from skeleton crops (.nii files)

        The programm loops over all subjects from the input (source) directory.
        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        # Creates target directory
        if not os.path.exists(self.tgt_dir):
            os.makedirs(self.tgt_dir)

        # Parallelization of mesh generation
        result = pqdm(
            self.list_subjects,
            self.create_one_mesh,
            n_jobs=define_njobs())
        # result = []
        # for sub in self.list_subjects:
        #     bucket = self.create_one_mesh(sub)
        #     result.append(bucket)
        buckets = dict(zip(self.list_subjects, result))
        return buckets


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='remove_hull.py',
        description='Generates meshes from skeleton crops')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the MRI data lies. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the cropped and pickle files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        "-l", "--subjects_list", type=list, default=None,
        help='python list containing subjects for whom creating meshes'
             'Default is : None')
    parser.add_argument(
        "-f", "--subjects_file", type=str, default=None,
        help='csv file containing subjects for whom creating meshes'
             '0 subject is allowed, for debug purpose.'
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)
    params['src_dir'] = args.src_dir
    params['tgt_dir'] = args.tgt_dir
    params['side'] = args.side
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


def main(argv):
    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        params = parse_args(argv)
        # Actual API
        dataset = DatasetHullRemoved(src_dir=params['src_dir'],
                                     tgt_dir=params['tgt_dir'],
                                     side=params['side'],
                                     number_subjects=params['nb_subjects'],
                                     list_subjects=None,
                                     file_subjects=None)

        dataset.create_meshes()

    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """
    main(argv=sys.argv[1:])
