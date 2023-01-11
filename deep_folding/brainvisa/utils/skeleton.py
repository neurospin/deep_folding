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

from deep_folding.brainvisa.utils.graph import create_empty_volume_from_graph
from deep_folding.config.logs import set_file_logger

# junction type 'wide' or 'thin'
from deep_folding.brainvisa.utils.constants import _JUNCTION_DEFAULT

# Defines logger
log = set_file_logger(__file__)

def generate_skeleton_thin_junction(
        graph: aims.Graph) -> aims.Volume:
    """Converts an aims graph into skeleton volumes

    It should produce thin junctions as vertices (aims_ss, aims_bottom)
    are written after edges (junction, plidepassage).
    Thus, when voxels are present in both, simple and bottom surfaces override
    junctions
    """
    vol_skel = create_empty_volume_from_graph(graph)
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
        graph: aims.Graph) -> aims.Volume:
    """Converts an aims graph into skeleton volumes

    It should produce wide junctions as edges (junction, plidepassage)
    are written after vertices (aims_ss, aims_bottom).
    Thus, when voxels are present in both, junction voxels override
    simple surface and bottom voxels
    """
    vol_skel = create_empty_volume_from_graph(graph)
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
