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

"Utilities to generate skeletons from graph"

import numpy as np
from soma import aims

from deep_folding.brainvisa import DeepFoldingError
from deep_folding.brainvisa.utils.graph import create_empty_volume_from_graph
from deep_folding.config.logs import set_file_logger

# junction type 'wide' or 'thin'
from deep_folding.brainvisa.utils.constants import _JUNCTION_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def is_skeleton(arr):
    """checks if arr is a skeleton"""
    skel_values = np.array([0, 30, 35, 60, 100, 110, 120])
    return np.isin(arr, skel_values).all()


def generate_skeleton_thin_junction(
        graph: aims.Graph, volume: aims.Volume) -> aims.Volume:
    """Converts an aims graph into skeleton volumes
    It should produce thin junctions as vertices (aims_ss, aims_bottom)
    are written after edges (junction, plidepassage).
    Thus, when voxels are present in both, simple and bottom surfaces override
    junctions
    """
    vol_skel = volume
    arr_skel = np.asarray(vol_skel)

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction': 110}.items():
            bucket = edge.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    for edge in graph.edges():
        if edge.getSyntax() == 'hull_junction':
            bucket = edge.get('aims_junction')
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = 35

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage': 120}.items():
            bucket = edge.get(bucket_name)
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
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    return vol_skel


def generate_skeleton_wide_junction(
        graph: aims.Graph, volume: aims.Volume) -> aims.Volume:
    """Converts an aims graph into skeleton volumes
    It should produce wide junctions as edges (junction, plidepassage)
    are written after vertices (aims_ss, aims_bottom).
    Thus, when voxels are present in both, junction voxels override
    simple surface and bottom voxels
    """
    vol_skel = volume
    arr_skel = np.asarray(vol_skel)

    cnt_duplicate = 0
    cnt_total = 0

    for vertex in graph.vertices():
        for bucket_name, value in {'aims_other': 100,
                                   'aims_ss': 60, 'aims_bottom': 30}.items():
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    # cnt_total += 1
                    # if arr_skel[i, j, k] != 0:
                    #     cnt_duplicate += 1
                    arr_skel[i, j, k] = value

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction': 110}.items():
            bucket = edge.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage': 120}.items():
            bucket = edge.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_skel[i, j, k] = value

    return vol_skel


def generate_skeleton_from_graph(
        graph: aims.Graph,
        junction: str = _JUNCTION_DEFAULT) -> aims.Volume:
    """Generates skeleton from graph"""
    volume = create_empty_volume_from_graph(graph)
    if junction == 'wide':
        vol_skel = generate_skeleton_wide_junction(graph, volume)
    else:
        vol_skel = generate_skeleton_thin_junction(graph, volume)
    return vol_skel


def generate_skeleton_from_graph_file(graph_file: str,
                                      skeleton_file: str,
                                      junction: str = _JUNCTION_DEFAULT):
    """Generates skeleton from graph file"""
    graph = aims.read(graph_file)
    vol_skeleton = generate_skeleton_from_graph(graph, junction)
    if not is_skeleton(np.asarray(vol_skeleton)):
        raise ValueError(f"{skeleton_file} has unexpected skeleton values")
    aims.write(vol_skeleton, skeleton_file)


def generate_full_skeleton(graph_file_left: str,
                           graph_file_right: str,
                           skeleton_file: str,
                           junction: str = _JUNCTION_DEFAULT):
    """Generates full skeleton from right and left graph files"""
    graph_left = aims.read(graph_file_left)
    graph_right = aims.read(graph_file_right)

    # Sanity check
    # TODO: find the good keys to check
    keys_to_check = ['voxel_size', 'transformations', 'referentials', 'referential']
    for k in keys_to_check:
        try:
            if graph_left[k] != graph_right[k]:
                raise DeepFoldingError(f"The attribute {k} is not the same in the right graph ({graph_right[k]}) "
                                       f"and in the left graph ({graph_left[k]})")
        except KeyError as e:
            log.warning(f"The attribute {e} is not in the graphs ({graph_file_right} or {graph_file_left})")
    # Get the dimensions for the new volume
    boundingbox_max_left = np.asarray(graph_left["boundingbox_max"])
    boundingbox_max_right = np.asarray(graph_right["boundingbox_max"])
    boundingbox_max = np.maximum(boundingbox_max_left, boundingbox_max_right)
    log.debug(f"Boundingbox max : {boundingbox_max}")

    # Create empty volumes with the new dimensions
    dimensions = (boundingbox_max[0] + 1,
                  boundingbox_max[1] + 1,
                  boundingbox_max[2] + 1,
                  1)
    empty_vol_left = create_empty_volume_from_graph(graph_left, dimensions=dimensions)
    empty_vol_right = create_empty_volume_from_graph(graph_right, dimensions=dimensions)
    vol_skeleton = create_empty_volume_from_graph(graph_right, dimensions=dimensions)
    # Generate the skeletons according to the junction
    if junction == 'wide':
        vol_skeleton_left = generate_skeleton_wide_junction(graph_left, empty_vol_left)
        vol_skeleton_right = generate_skeleton_wide_junction(graph_right, empty_vol_right)
    else:
        vol_skeleton_left = generate_skeleton_thin_junction(graph_left, empty_vol_left)
        vol_skeleton_right = generate_skeleton_thin_junction(graph_right, empty_vol_right)

    priority_order = {0: 15, 10: 10, 11: 14, 20: 9, 30: 12, 35: 11, 40: 8,
                      50: 7, 60: 13, 70: 6, 80: 5, 90: 4, 100: 3, 110: 2, 120: 1}
    arr_skeleton_left = np.asarray(vol_skeleton_left)
    arr_skeleton_right = np.asarray(vol_skeleton_right)
    arr_skeleton = np.asarray(vol_skeleton)

    # Add the left and right skeletons
    # For contentious voxels (voxels which have two different values in the two skeletons),
    # the value is chosen according to the priority order
    vectorize = np.vectorize(lambda x: priority_order[x])
    mask = vectorize(arr_skeleton_right) >= vectorize(arr_skeleton_left)
    arr_skeleton[mask] = arr_skeleton_left[mask]
    arr_skeleton[~mask] = arr_skeleton_right[~mask]
    arr_skeleton = arr_skeleton.astype(int)

    # Sanity checks
    # FIXME : select good threshold
    threshold = 200
    nb_contentious_voxels = np.count_nonzero(np.logical_and(arr_skeleton_left, arr_skeleton_right))
    log.debug(f"Unique value of the skeleton : {np.unique(arr_skeleton)}")
    log.debug(f"Number of conflict voxels between left and right skeletons : {nb_contentious_voxels}")
    if nb_contentious_voxels > threshold:
        log.warning(f"Left and right graph files have {nb_contentious_voxels} voxels with different values ! "
                    f"Graph files : {graph_file_left} and {graph_file_left}")

    if not is_skeleton(np.asarray(vol_skeleton)):
        raise DeepFoldingError(f"{skeleton_file} has unexpected skeleton values")
    aims.write(vol_skeleton, skeleton_file)
