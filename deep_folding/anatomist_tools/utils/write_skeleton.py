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
        prog='convert_volume_to_bucket.py',
        description='Generates bucket files converted from volume files')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source directory where the graph data lies.')
    parser.add_argument(
        "-t", "--tgt_dir", type=str, required=True,
        help='Output directory where to put skeleton files.')
    parser.add_argument(
        "-i", "--side", type=str, required=True,
        help='Hemisphere side (either L or R).')

    args = parser.parse_args(argv)

    return args


class GraphConvert2Skeleton:
    """
    """
    def __init__(self, src_dir, tgt_dir, side):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.side = side
        self.graph_dir = "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/"

    def write_skeleton(self, subject):
        """
        """
        graph_file = f"{self.side}{subject}_default_session_auto.arg"
        graph = aims.read(os.path.join(self.src_dir, f"{subject}/"+self.graph_dir, graph_file))

        skeleton_filename = f"{self.tgt_dir}/{self.side}skeleton_{subject}_generated.nii.gz"

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


    def write_loop(self):
        filenames = glob.glob(f"{self.src_dir}/*/")
        list_subjects = [re.search('([ae\d]{5,6})', filename).group(1) for filename in filenames]
        pqdm(list_subjects, self.write_skeleton, n_jobs=define_njobs())


def main(argv):
    """
    """
    # Parsing arguments
    args = parse_args(argv)
    conversion = GraphConvert2Skeleton(args.src_dir, args.tgt_dir, args.side)
    conversion.write_loop()


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
