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
  >>> python generate_skeletons.py


"""

import argparse
import glob
import logging
import re
import sys
from os.path import abspath
from os.path import basename
from typing import Tuple

import numpy as np
from deep_folding.brainvisa import _ALL_SUBJECTS
from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from pqdm.processes import pqdm
from deep_folding.config.logs import set_file_logger
from soma import aims

# Defines logger
log = set_file_logger(__file__)

# Default directory in which lies the dataset
_SRC_DIR_DEFAULT = "/tgcc/hcp/ANALYSIS/3T_morphologist"

# Default directory where we put skeletons
_SKELETON_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/hcp"

# Gives the relative path to the manually labelled graph .arg
# in the supervised database
_PATH_TO_GRAPH_DEFAULT = "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_*"

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'R'

# junction type 'wide' or 'thin'
_JUNCTION_DEFAULT = 'thin'

def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description='Generates skeleton files from graphs')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the graph data lies. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_SKELETON_DIR_DEFAULT,
        help='Output directory where to put skeleton files.'
            'Default is : ' + _SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-p", "--path_to_graph", type=str,
        default=_PATH_TO_GRAPH_DEFAULT,
        help='Relative path to graph. '
             'Default is ' + _PATH_TO_GRAPH_DEFAULT)
    parser.add_argument(
        "-j", "--junction", type=str, default=_JUNCTION_DEFAULT,
        help='junction rendering (either \'wide\' or \'thin\') '
             f"Default is {_JUNCTION_DEFAULT}")
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
        'If no option is provided then logging.INFO is selected. '
        'If one option -v (or -vv) or more is provided '
        'then logging.DEBUG is selected.')

    args = parser.parse_args(argv)

    setup_log(args,
              log_dir=f"{args.output_dir}/skeletons",
              prog_name=basename(__file__),
              suffix='right' if args.side == 'R' else 'left')

    params = {}

    params['src_dir'] = abspath(args.src_dir)
    params['skeleton_dir'] = abspath(args.output_dir)
    params['path_to_graph'] = args.path_to_graph
    params['side'] = args.side
    params['junction'] = args.junction
    params['parallel'] = args.parallel
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


def create_empty_volume_from_graph(graph: aims.Graph) -> aims.Volume:
    """Creates empty volume with graph header"""

    voxel_size = graph['voxel_size'][:3]
    # Adds 1 for each x,y,z dimension
    dimensions = [i + j for i,
                  j in zip(graph['boundingbox_max'], [1, 1, 1, 0])]

    vol = aims.Volume(dimensions, dtype='S16')
    vol.header()['voxel_size'] = voxel_size
    if 'transformations' in graph.keys():
        vol.header()['transformations'] = graph['transformations']
    if 'referentials' in graph.keys():
        vol.header()['referentials'] = graph['referentials']
    if 'referential' in graph.keys():
        vol.header()['referential'] = graph['referential']
    return vol


def generate_skeleton_thin_junction(
        graph: aims.Graph) -> aims.Volume:
    """Converts an aims graph into skeleton and foldlabel volumes

    It should produce thin junctions as vertices (aims_ss, aims_bottom)
    are written after edges (junction, plidepassage).
    Thus, when voxels are present in both, simple and bottom surfaces override
    junctions
    """
    vol_skel = create_empty_volume_from_graph(graph)
    arr_skel = np.asarray(vol_skel)

    # Sorted in ascendent priority
    label = {'aims_other': 1,
             'aims_ss': 1000,
             'aims_bottom': 2000,
             'aims_junction': 3000,
             'aims_plidepassage': 4000}

    cnt_duplicate = 0
    cnt_total = 0

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction': 110}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage': 120}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    for vertex in graph.vertices():
        for bucket_name, value in {'aims_other': 100,
                                   'aims_ss': 60, 'aims_bottom': 30}.items():
            bucket = vertex.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    cnt_total += 1
                    if arr_skel[i, j, k] != 0:
                        cnt_duplicate += 1
                    arr_skel[i, j, k] = value

    return vol_skel


def generate_skeleton_wide_junction(
        graph: aims.Graph) -> aims.Volume:
    """Converts an aims graph into skeleton and foldlabel volumes

    It should produce wide junctions as edges (junction, plidepassage)
    are written after vertices (aims_ss, aims_bottom).
    Thus, when voxels are present in both, junction voxels override
    simple surface and bottom voxels
    """
    vol_skel = create_empty_volume_from_graph(graph)
    arr_skel = np.asarray(vol_skel)

    # Sorted in ascendent priority
    label = {'aims_other': 1,
             'aims_ss': 1000,
             'aims_bottom': 2000,
             'aims_junction': 3000,
             'aims_plidepassage': 4000}

    cnt_duplicate = 0
    cnt_total = 0

    for vertex in graph.vertices():
        for bucket_name, value in {'aims_other': 100,
                                   'aims_ss': 60, 'aims_bottom': 30}.items():
            bucket = vertex.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    cnt_total += 1
                    if arr_skel[i, j, k] != 0:
                        cnt_duplicate += 1
                    arr_skel[i, j, k] = value

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction': 110}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage': 120}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    return vol_skel


def generate_skeleton_from_graph(graph: aims.Graph,
                                 junction: str = _JUNCTION_DEFAULT) -> aims.Volume:
    """Generates skeleton from graph"""
    if junction == 'wide':
        vol_skel = generate_skeleton_wide_junction(graph)
    else:
        vol_skel = generate_skeleton_thin_junction(graph)
    return vol_skel


def generate_skeleton_from_graph_file(graph_file: str,
                                      skeleton_file: str,
                                      junction: str = _JUNCTION_DEFAULT):
    """Generates skeleton from graph file"""
    graph = aims.read(graph_file)
    vol_skeleton = generate_skeleton_from_graph(graph, junction)
    aims.write(vol_skeleton, skeleton_file)


class GraphConvert2Skeleton:
    """Class to convert all graphs from a folder into skeletons

    It contains all information to scan a dataset for graphs
    and writes skeletons and foldlabels into target directory
    """

    def __init__(self, src_dir, skeleton_dir,
                 side, junction, parallel,
                 path_to_graph):
        self.src_dir = src_dir
        self.skeleton_dir = skeleton_dir
        self.side = side
        self.junction = junction
        self.parallel = parallel
        self.path_to_graph = path_to_graph
        self.skeleton_dir = f"{self.skeleton_dir}/skeletons/{self.side}"
        create_folder(abspath(self.skeleton_dir))

    def generate_one_skeleton(self, subject: str):
        """Generates and writes skeleton for one subject.
        """
        graph_file = glob.glob(
            f"{self.src_dir}/{subject}*/{self.path_to_graph}/{self.side}{subject}*.arg")[0]
        skeleton_file = f"{self.skeleton_dir}/{self.side}skeleton_generated_{subject}.nii.gz"

        generate_skeleton_from_graph_file(graph_file,
                                          skeleton_file,
                                          self.junction)

    def compute(self, number_subjects):
        """Loops over subjects and converts graphs into skeletons.
        """
        # Gets list fo subjects
        filenames = glob.glob(f"{self.src_dir}/*/")
        list_subjects = [
            re.search(
                '([ae\\d]{5,6})',
                filename).group(0) for filename in filenames]
        list_subjects = select_subjects_int(list_subjects, number_subjects)

        # Performs computation on all subjects either serially or in parallel
        if self.parallel:
            log.info(
                "PARALLEL MODE: subjects are computed in parallel.")
            pqdm(list_subjects, self.generate_one_skeleton, n_jobs=define_njobs())
        else:
            log.info(
                "SERIAL MODE: subjects are scanned serially, "
                "without parallelism")
            for sub in list_subjects:
                self.generate_one_skeleton(sub)


def generate_skeletons(
        src_dir=_SRC_DIR_DEFAULT,
        skeleton_dir=_SKELETON_DIR_DEFAULT,
        path_to_graph=_PATH_TO_GRAPH_DEFAULT,
        side=_SIDE_DEFAULT,
        junction=_JUNCTION_DEFAULT,
        parallel=False,
        number_subjects=_ALL_SUBJECTS):
    """Generates skeletons from graphs"""

    # Initialization
    conversion = GraphConvert2Skeleton(
        src_dir=src_dir,
        skeleton_dir=skeleton_dir,
        path_to_graph=path_to_graph,
        side=side,
        junction=junction,
        parallel=parallel
    )
    # Actual generation of skeletons from graphs
    conversion.compute(number_subjects=number_subjects)


@exception_handler
def main(argv):
    """Reads argument line and generates skeleton from graph

    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    generate_skeletons(
        src_dir=params['src_dir'],
        skeleton_dir=params['skeleton_dir'],
        path_to_graph=params['path_to_graph'],
        side=params['side'],
        junction=params['junction'],
        parallel=params['parallel'],
        number_subjects=params['nb_subjects'])


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
