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

"""Write skeletons from graph files

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python write_skeleton.py


"""

import sys
import glob
import os
import argparse
from tqdm import tqdm
from soma import aims
import numpy as np
import dico_toolbox as dtx
from convert_volume_to_bucket import get_basename_without_extension

_SIDE = 'R'
test_mode = True

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
        help='Source directory where the graph data lies.')
    parser.add_argument(
        "-t", "--tgt_dir", type=str, required=True,
        help='Output directory where to put skeleton files.')

    args = parser.parse_args(argv)

    return args


def build_skeleton_filename(subject, tgt_dir):
    """Returns bucket filename"""
    return f"{tgt_dir}/{_SIDE}skeleton_{subject}.nii.gz"


def loop_over_directory(src_dir, tgt_dir, test_mode=test_mode):
    """Loops conversion over input directory
    """
    # Gets and creates all filenames
    if test_mode:
        src_dir = '/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist'
        graph_dir = 't1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto'
    else :
        graph_dir = 't1mri/default_acquisition/default_analysis/folds/3.1/default_session_manual'

    #filenames = glob.glob(f"{src_dir}/*/{graph_dir}/{_SIDE}*.arg")
    #print(filenames)
    filenames = ['/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist/299760/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/R299760_default_session_auto.arg',
                 '/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist/100307/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/R100307_default_session_auto.arg']

    subjects = [get_basename_without_extension(filename) for filename in filenames]
    #subjects = ['299760', '100206', '100307']
    skeleton_filenames = [build_skeleton_filename(subject, tgt_dir) for subject in subjects]
    print(skeleton_filenames)
    for graph_filename, skeleton_filename in tqdm(zip(filenames, skeleton_filenames), total=len(filenames)):
        write_skeleton(graph_filename, skeleton_filename)


def write_skeleton(graph_filename, skeleton_filename):
    """
    """
    graph = aims.read(graph_filename)
    voxel_size = graph['voxel_size'][:3]

    dimensions = [i+j for i, j in zip(graph['boundingbox_max'], [1,1,1,0])]
    vol = aims.Volume(dimensions, dtype='S16')
    vol.header()['voxel_size'] = voxel_size
    if 'transformations' in graph.keys():
        vol.header()['transformations'] = graph['transformations']
    if 'referentials' in graph.keys():
        vol.header()['referentials'] = graph['referentials']
    if 'referential' in graph.keys():
        vol.header()['referential'] = graph['referential']
    arr = np.asarray(vol)

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction':110, 'aims_plidepassage':120}.items():
            bucket = edge.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    cnt += 1
                    arr[i,j,k] = value

    for vertex in graph.vertices():
        for bucket_name, value in {'aims_bottom': 30, 'aims_ss': 60, 'aims_other': 100}.items():
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    arr[i,j,k] = value

    aims.write(vol, skeleton_filename)


def main(argv):
    """
    """
    # Parsing arguments
    args = parse_args(argv)
    loop_over_directory(args.src_dir, args.tgt_dir)


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
