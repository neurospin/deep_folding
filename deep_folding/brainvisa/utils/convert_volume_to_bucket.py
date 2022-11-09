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
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the

"""
This program converts volumes contained in a folder into buckets.
It writes bucket files in the output folder
"""
import argparse
import glob
import os
import sys
import numpy as np
import dico_toolbox as dtx

import six
from deep_folding.brainvisa.utils.remove_hull import convert_volume_to_bucket
from deep_folding.brainvisa.utils.folder import create_folder
from soma import aims
from tqdm import tqdm


def read_convert_write(vol_filename, bucket_filename, mask_dir=None,
                        distmap=False, mesh=False):
    """Reads volume, converts and writes back bucket.

    Args:
        vol_filename [str]: path to input volume file
        bucket_filename [str]: path to output bucket file
    """
    vol = aims.read(vol_filename)
    if mask_dir:
        vol_arr = np.asarray(vol)
        mask = aims.read(mask_dir)
        mask_arr = np.asarray(mask)
        # Apply mask
        vol_arr[mask_arr==0] = 0

        if distmap:
            # Apply mask
            vol_arr[mask_arr==0] = 10
            # Threshold distmap
            vol_arr[vol_arr<=1] = 1
            vol_arr[vol_arr>1] = 0

        bucket_map = dtx.convert.volume_to_bucketMap_aims(vol_arr)
        if mesh:
            bucket_map = dtx.convert.bucket_to_mesh(bucket_map[0])

    else:
        bucket_map, _ = convert_volume_to_bucket(vol)

    aims.write(bucket_map, bucket_filename)


def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='convert_volume_to_bucket.py',
        description='Generates bucket files converted from volume files')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source directory where the MRI data lies.')
    parser.add_argument(
        "-t", "--tgt_dir", type=str, required=True,
        help='Output directory where to put bucket files.')
    parser.add_argument(
        "-m", "--mask_dir", type=str, required=False,
            help='Mask directory.')
    parser.add_argument(
        "-b", "--mesh", type=bool, required=False,
            help='Whether to output buckets or meshes.')
    parser.add_argument(
        "-d", "--distmap", type=bool, required=False,
            help='Type of input: skeleton or distmap.')

    args = parser.parse_args(argv)

    return args


def get_basename_without_extension(filename):
    """Returns file basename without extension"""
    basename = os.path.basename(filename)
    
    without_extension = basename.split('.')[0]
    return without_extension


def build_bucket_filename(subject, tgt_dir, mesh):
    """Returns bucket filename"""
    if mesh:
        return f"{tgt_dir}/{subject}.mesh"
    else:
        return f"{tgt_dir}/{subject}.bck"

def loop_over_directory(src_dir, tgt_dir, mask_dir, mesh, distmap):
    """Loops conversion over input directory
    """
    # Gets and creates all filenames
    filenames = glob.glob(f"{src_dir}/*.nii.gz")
    subjects = [get_basename_without_extension(filename) 
                for filename in filenames]
    bucket_filenames = [
        build_bucket_filename(
            subject,
            tgt_dir,
            mesh) for subject in subjects]
    create_folder(tgt_dir)

    # Creates target d    # python3 convert_volume_to_bucket.py \
    # -s /neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/simple_combined/Rcrops \
    # -t /neurospin/di
    # Makes the actual conversion
    for vol_filename, bucket_filename in tqdm(
            zip(filenames, bucket_filenames), total=len(filenames)):
        read_convert_write(vol_filename=vol_filename,
                           bucket_filename=bucket_filename,
                           mask_dir=mask_dir,
                           mesh=mesh,
                           distmap=distmap)


def main(argv):
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        args = parse_args(argv)
        loop_over_directory(args.src_dir, args.tgt_dir, args.mask_dir,
                            args.mesh, args.distmap)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])

    # Example of use:
    # python3 convert_volume_to_bucket.py \
    # -s /neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/simple_combined/Rcrops \
    # -t /neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/simple_combined/Rbuckets
