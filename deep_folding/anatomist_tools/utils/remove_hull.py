# -*- coding: utf-8 -*-
# /usr/bin/env python3.6 + brainvisa compliant env
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
import moving_averages as ma
import colorado as cld

# pyAims import
from soma import aims

# system imports
import os
import glob
import numpy as np
import argparse
import sys

from pqdm.processes import pqdm
from joblib import cpu_count

_ALL_SUBJECTS = -1

# Input directory contaning the morphologist analysis of the HCP database
_SRC_DIR_DEFAULT = '/nfs/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm/'

# Directory containing subjects meshes without hull once created
# default corresponds to
# -------------------------
_TGT_DIR_DEFAULT = '/nfs/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm/meshes/'

_SIDE_DEFAULT = 'R'

def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus-2, 1)

class DatasetHullRemoved:
    """Generates meshes of crops removing hull
    """

    def __init__(self, src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 side=_SIDE_DEFAULT):
        """Inits with list of directories and side

        Args:
            src_dir: string naming full path source directory,
                    containing crops images
            tgt_dir: name of target (output) directory with full path
            side: hemisphere side (either L for left, or R for right hemisphere)
        """
        self.side = side
        self.src_dir = os.path.join(src_dir, f"{self.side}crops")
        self.tgt_dir = tgt_dir

    def remove_hull(self, padding, ext):
        """Removes the pixels on the hull
        Pixels on the hull are defined as being in contact with both the internal and external part

        Args:
            arr: numpy array modified in the array
            padding: padding of the image, equal to the extension ext
            ext: local array extension in which to look for external and internal pixels
        """
        arr_pad = np.pad(self.arr, ((padding,padding), (padding,padding), (padding,padding), (0,0)), 'constant', constant_values=0)
        EXTERNAL = 11 # Value of external part
        INTERNAL = 0 # Value of the internal part of the brain
        l = 0 # last t dimension
        coords = np.argwhere(arr_pad > EXTERNAL)

        for i,j,k,l in coords:
            if arr_pad[i,j,k,l] != EXTERNAL and arr_pad[i,j,k,l] != INTERNAL:
                local_array = arr_pad[i-ext:i+ext+1,j-ext:j+ext+1,k-ext:k+ext+1,l]
                if np.any(local_array == 0) and np.any(local_array == 11):
                    self.arr[i-padding,j-padding,k-padding,l] = 0

    def threshold_and_binarize(self, threshold):
        """Threshold images"""
        self.arr[np.where(self.arr < threshold)]= 0
        self.arr[np.where(self.arr >= threshold)]= 32767
        self.arr = aims.Volume(self.arr)

    def create_one_mesh(self, subject_id):
        """Creates one mesh from a skeleton crop (.nii file)

        Args:
            subject_id: string giving the subject ID
        """
        c = aims.Converter_rc_ptr_Volume_S16_BucketMap_VOID()

        # Constant definition
        padding = 1
        ext = 1 # extent (equal to padding)
        threshold = 12 # 30 if to keep only bottom values

        # Names directory where subject analysis files are stored
        subject_dir = \
            os.path.join(self.src_dir, f"{subject_id}_normalized.nii.gz")

        # Reads nifti as AIMS volume and conversion to array
        vol = aims.read(subject_dir)
        self.arr = np.asarray(vol)

        # Removes hull
        self.remove_hull(padding, ext)

        # Thresholds and "binarizes"
        self.threshold_and_binarize(threshold)

        # Conversion of volume to bucket
        bucket = c(self.arr)

        # Conversion of bucket to mesh
        m = cld.bucket_to_mesh(bucket[0])

        # Writing of the mesh in tgt_dir folder
        aims.write(m, f"{self.tgt_dir}mesh_{subject_id}.gii")

    def create_meshes(self, number_subjects=_ALL_SUBJECTS):
        """Creates meshes from skeleton crops (.nii files)

        The programm loops over all subjects from the input (source) directory.
        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """
        if number_subjects:
            # subjects are detected as the directory names under src_dir
            list_all_subjects = [dI[:6] for dI in os.listdir(self.src_dir)\
             if os.path.isdir(self.src_dir) and 'minf' not in dI]

            # Gives the possibility to list only the first number_subjects
            list_subjects = (
                list_all_subjects
                if number_subjects == _ALL_SUBJECTS
                else list_all_subjects[:number_subjects])

            # Creates target directory
            if not os.path.exists(self.tgt_dir):
                os.makedirs(self.tgt_dir)

            pqdm(list_subjects, self.create_one_mesh, n_jobs=define_njobs())


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
        dataset = DatasetHullRemoved(src_dir=_SRC_DIR_DEFAULT,
                                     tgt_dir=_TGT_DIR_DEFAULT,
                                     side=_SIDE_DEFAULT)
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
