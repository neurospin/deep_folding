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
import re
import argparse
from pqdm.processes import pqdm
from joblib import cpu_count
from soma import aims
import numpy as np


def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus-2, 1)

def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='write_skeleton.py',
        description='Generates skeleton and foldlabel files from graphs')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source directory where the graph data lies.')
    parser.add_argument(
        "-t", "--tgt_dir", type=str, required=True,
        help='Output directory where to put skeleton and fold label files.')
    parser.add_argument(
        "-i", "--side", type=str, required=True,
        help='Hemisphere side (either L or R).')
    parser.add_argument(
        "-n", "--nb_subjects", type=int, required=False, default=5,
        help='Number of subjects')
    parser.add_argument(
        "-v", "--verbose", type=bool, required=False,
        help='If verbose is true, no parallelism.')

    args = parser.parse_args(argv)

    return args

def create_volume_from_graph(graph):
    """Creates empty volume with graph header"""

    voxel_size = graph['voxel_size'][:3]
    # Adds 1 for each x,y,z dimension
    dimensions = [i+j for i, j in zip(graph['boundingbox_max'], [1,1,1,0])]

    vol = aims.Volume(dimensions, dtype='S16')
    vol.header()['voxel_size'] = voxel_size
    if 'transformations' in graph.keys():
        vol.header()['transformations'] = graph['transformations']
    if 'referentials' in graph.keys():
        vol.header()['referentials'] = graph['referentials']
    if 'referential' in graph.keys():
        vol.header()['referential'] = graph['referential']
    return vol

class GraphConvert2Skeleton:
    """
    """
    def __init__(self, src_dir, tgt_dir, side):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.side = side
        self.graph_subdir = "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_*"

    def write_skeleton(self, subject):
        """
        """
        # graph_file = f"{self.side}{subject}*.arg"
        graph_file = glob.glob(f"{self.src_dir}/{subject}*/{self.graph_subdir}/{self.side}{subject}*.arg")[0]
        graph = aims.read(graph_file)

        skeleton_filename = f"{self.tgt_dir}/skeleton/{self.side}/{self.side}skeleton_generated_{subject}.nii.gz"
        vol_skel = create_volume_from_graph(graph)
        arr_skel = np.asarray(vol_skel)

        foldlabel_filename = f"{self.tgt_dir}/foldlabel/{self.side}/{self.side}foldlabel_{subject}.nii.gz"
        vol_label = create_volume_from_graph(graph)
        arr_label = np.asarray(vol_label)

        # label = {'aims_ss':0,
        #          'aims_bottom': 1000,
        #          'aims_other': 2000,
        #          'aims_junction': 3000,
        #          'aims_plidepassage': 4000}

        # Sorted in ascendent priority
        # label = {'aims_other':0,
        #          'aims_ss': 1000,
        #          'aims_bottom': 2000,
        #          'aims_junction': 3000,
        #          'aims_plidepassage': 4000}

        # Sorted in descendent priority
        label = {'aims_other': 4000,
                 'aims_ss': 3000,
                 'aims_bottom': 2000,
                 'aims_junction': 2000,
                 'aims_plidepassage': 0}

        for edge in graph.edges():
            for bucket_name, value in {'aims_junction':110, 'aims_plidepassage':120}.items():
                bucket = edge.get(bucket_name)
                label[bucket_name] += 1
                if bucket is not None:
                    voxels = np.array(bucket[0].keys())
                    if voxels.shape == (0,):
                        continue
                    for i,j,k in voxels:
                        arr_skel[i,j,k] = value
                        arr_label[i,j,k] = label[bucket_name]

        for vertex in graph.vertices():
            for bucket_name, value in {'aims_bottom': 30, 'aims_ss': 60, 'aims_other': 100}.items():
                bucket = vertex.get(bucket_name)
                label[bucket_name] += 1
                if bucket is not None:
                    voxels = np.array(bucket[0].keys())
                    if voxels.shape == (0,):
                        continue
                    for i,j,k in voxels:
                        arr_skel[i,j,k] = value
                        arr_label[i,j,k] = label[bucket_name]

        aims.write(vol_skel, skeleton_filename)
        aims.write(vol_label, foldlabel_filename)


    def write_loop(self, nb_subjects, verbose=False):
        filenames = glob.glob(f"{self.src_dir}/*/")
        list_subjects = [re.search('([ae\d]{5,6})', filename).group(0) for filename in filenames]
        if verbose:
            for sub in list_subjects[:nb_subjects]:
                self.write_skeleton(sub)
        else:
            pqdm(list_subjects, self.write_skeleton, n_jobs=define_njobs())


def main(argv):
    """
    """
    # Parsing arguments
    args = parse_args(argv)
    conversion = GraphConvert2Skeleton(args.src_dir, args.tgt_dir, args.side)
    conversion.write_loop(nb_subjects=args.nb_subjects,
                          verbose=args.verbose)


if __name__ == '__main__':
    src_dir = "/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist"
    tgt_dir = "/neurospin/dico/data/deep_folding/datasets/hcp"
    args = "-i R -v True -n 5 -s " + src_dir + " -t " + tgt_dir
    argv = args.split(' ')
    main(argv=argv)

    # This permits to call main also from another python program
    # without having to make system calls
    # main(argv=sys.argv[1:])
