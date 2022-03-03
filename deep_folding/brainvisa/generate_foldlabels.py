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
  >>> python generate_skeleton.py


"""

import sys
import glob
import re
import argparse
import logging
import numpy as np
from pqdm.processes import pqdm
from os.path import abspath
from os.path import basename
from soma import aims
from typing import Tuple

from deep_folding.anatomist_tools.utils.list import get_sublist
from deep_folding.anatomist_tools.utils.folder import create_folder
from deep_folding.anatomist_tools.utils.logs import log_command_line
from deep_folding.anatomist_tools.utils.parallel import define_njobs

logging.basicConfig(level = logging.INFO)

log = logging.getLogger(basename(__file__))

_ALL_SUBJECTS = -1

def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='generate_skeleton.py',
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
        "-j", "--junction", type=str, required=True,
        help='junction rendering (either \'wide\' or \'thin\')')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        "-v", "--verbose",
        default=False,
        action='store_true',
        help='If verbose is true, no parallelism.')

    args = parser.parse_args(argv)
    args.src_dir = abspath(args.src_dir)
    args.tgt_dir = abspath(args.tgt_dir)

    return args


def create_volume_from_graph(graph: aims.Graph) -> aims.Volume:
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


def generate_skeleton_thin_junction(graph: aims.Graph) -> Tuple[aims.Volume, aims.Volume]:
    """Converts an aims graph into skeleton and foldlabel volumes

    It should produce thin junctions as vertices (aims_ss, aims_bottom)
    are written after edges (junction, plidepassage). 
    Thus, when voxels are present in both, simple and bottom surfaces override
    junctions
    """
    vol_skel = create_volume_from_graph(graph)
    arr_skel = np.asarray(vol_skel)

    vol_label = create_volume_from_graph(graph)
    arr_label = np.asarray(vol_label)

    # Sorted in ascendent priority
    label = {'aims_other':1,
             'aims_ss': 1000,
             'aims_bottom': 2000,
             'aims_junction': 3000,
             'aims_plidepassage': 4000}

    cnt_duplicate = 0
    cnt_total = 0

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction':110}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    arr_skel[i,j,k] = value
                    arr_label[i,j,k] = label[bucket_name]

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage':120}.items():
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
        for bucket_name, value in {'aims_other': 100, 'aims_ss': 60, 'aims_bottom': 30}.items():
            bucket = vertex.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    cnt_total += 1
                    if arr_skel[i,j,k] != 0:
                        cnt_duplicate += 1
                    arr_skel[i,j,k] = value
                    arr_label[i,j,k] = label[bucket_name]

    return vol_skel, vol_label


def generate_skeleton_wide_junction(graph: aims.Graph) -> Tuple[aims.Volume, aims.Volume]:
    """Converts an aims graph into skeleton and foldlabel volumes

    It should produce wide junctions as edges (junction, plidepassage)
    are written after vertices (aims_ss, aims_bottom). 
    Thus, when voxels are present in both, junction voxels override 
    simple surface and bottom voxels
    """
    vol_skel = create_volume_from_graph(graph)
    arr_skel = np.asarray(vol_skel)

    vol_label = create_volume_from_graph(graph)
    arr_label = np.asarray(vol_label)

    # Sorted in ascendent priority
    label = {'aims_other':1,
             'aims_ss': 1000,
             'aims_bottom': 2000,
             'aims_junction': 3000,
             'aims_plidepassage': 4000}

    cnt_duplicate = 0
    cnt_total = 0

    for vertex in graph.vertices():
        for bucket_name, value in {'aims_other': 100, 'aims_ss': 60, 'aims_bottom': 30}.items():
            bucket = vertex.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    cnt_total += 1
                    if arr_skel[i,j,k] != 0:
                        cnt_duplicate += 1
                    arr_skel[i,j,k] = value
                    arr_label[i,j,k] = label[bucket_name]

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction':110}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    arr_skel[i,j,k] = value
                    arr_label[i,j,k] = label[bucket_name]

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage':120}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i,j,k in voxels:
                    arr_skel[i,j,k] = value
                    arr_label[i,j,k] = label[bucket_name]

    return vol_skel, vol_label


class GraphConvert2Skeleton:
    """Class to convert graph into skeleton and foldlabel files

    It contains all information to scan a dataset for graphs
    and writes skeletons and foldlabels into target directory
    """
    def __init__(self, src_dir, tgt_dir, nb_subjects, side, junction):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.nb_subjects = nb_subjects
        self.side = side
        self.junction = junction
        self.graph_subdir = "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_*"
        self.skeleton_dir = f"{self.tgt_dir}/skeleton/{self.side}"
        self.foldlabel_dir = f"{self.tgt_dir}/foldlabel/{self.side}"

        create_folder(abspath(self.skeleton_dir))
        create_folder(abspath(self.foldlabel_dir))

    def generate_skeleton(self, subject: str):
        """Generates and writes skeleton for one subject.
        """
        graph_file = glob.glob(f"{self.src_dir}/{subject}*/{self.graph_subdir}/{self.side}{subject}*.arg")[0]
        graph = aims.read(graph_file)

        if self.junction == 'wide':
            vol_skel, vol_label = generate_skeleton_wide_junction(graph)
        else:
            vol_skel, vol_label = generate_skeleton_thin_junction(graph)

        skeleton_filename = f"{self.skeleton_dir}/{self.side}skeleton_generated_{subject}.nii.gz"
        foldlabel_filename = f"{self.foldlabel_dir}/{self.side}foldlabel_{subject}.nii.gz"

        aims.write(vol_skel, skeleton_filename)
        aims.write(vol_label, foldlabel_filename)


    def loop(self, nb_subjects, verbose=False):
        """Loops over subjects and converts graphs into skeletons.
        """
        filenames = glob.glob(f"{self.src_dir}/*/")
        list_subjects = [re.search('([ae\d]{5,6})', filename).group(0) for filename in filenames]
        list_subjects = get_sublist(list_subjects, nb_subjects)
        if verbose:
            log.info("VERBOSE MODE: subjects are scanned serially, without parallelism")
            for sub in list_subjects:
                self.generate_skeleton(sub)
        else:
            pqdm(list_subjects, self.generate_skeleton, n_jobs=define_njobs())


def main(argv):
    """
    """
    # Parsing arguments
    args = parse_args(argv)

    # Writes command line argument to target dir for logging
    log_command_line(args, "generate_skeleton.py", args.tgt_dir)

    # Converts graph to skeleton and foldlabel
    conversion = GraphConvert2Skeleton(args.src_dir,
                                       args.tgt_dir,
                                       args.nb_subjects,
                                       args.side,
                                       args.junction)
    conversion.loop(nb_subjects=args.nb_subjects,
                    verbose=args.verbose)


if __name__ == '__main__':
    # src_dir = "/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist"
    # tgt_dir = "/neurospin/dico/data/deep_folding/datasets/hcp"
    # args = "-i R -v True -n 5 -s " + src_dir + " -t " + tgt_dir
    # argv = args.split(' ')
    # main(argv=argv)

    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
