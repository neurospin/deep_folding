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

"Utilities to generate foldlabels from graph"

import numpy as np
from soma import aims

from deep_folding.brainvisa.utils.graph import create_empty_volume_from_graph
from deep_folding.config.logs import set_file_logger

# junction type 'wide' or 'thin'
from deep_folding.brainvisa.utils.constants import _JUNCTION_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def generate_foldlabel_thin_junction(
        graph: aims.Graph) -> aims.Volume:
    """Converts an aims graph into skeleton and foldlabel volumes

    It should produce thin junctions as vertices (aims_ss, aims_bottom)
    are written after edges (junction, plidepassage).
    Thus, when voxels are present in both, simple and bottom surfaces override
    junctions
    """

    vol_label = create_empty_volume_from_graph(graph)
    arr_label = np.asarray(vol_label)

    # Sorted in ascendent priority
    val_aims_ss = 1000
    add_val = {'aims_other': -1000,
               'aims_ss': 0,
               'aims_top': 7000,
               'aims_bottom': 6000,
               'aims_junction': 5000,
               'aims_plidepassage': 4000}

    for vertex in graph.vertices():
        val_aims_ss += 1

        for edge in range(len(vertex.edges())):
            if 'aims_plidepassage' in vertex.edges()[edge]:
                voxels_plidepassage = np.array(
                    vertex.edges()[edge]['aims_plidepassage'][0].keys())
                if voxels_plidepassage.shape == (0,):
                    continue
                for i, j, k in voxels_plidepassage:
                    if arr_label[i, j, k] > 2000:
                        arr_label[i, j, k] = val_aims_ss + \
                            add_val['aims_plidepassage']

        for bucket_name, value in {'aims_bottom': 6000,
                                   'aims_other': -1000, 'aims_ss': 0}.items():
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                for edge in range(len(vertex.edges())):
                    if 'aims_junction' in vertex.edges()[edge]:
                        voxels_junction = np.array(
                            vertex.edges()[edge]['aims_junction'][0].keys())
                        if voxels_junction.shape == (0,):
                            continue
                        for i, j, k in voxels_junction:
                            if arr_label[i, j, k] == 0:
                                arr_label[i, j, k] = val_aims_ss + \
                                    add_val['aims_junction']

                    e = vertex.edges()[edge]
                    if e.getSyntax() == 'hull_junction':
                        if 'aims_junction' in vertex.edges()[edge]:
                            voxels_junction = np.array(
                                vertex.edges()[edge]['aims_junction'][0].keys()
                                )
                            if voxels_junction.shape == (0,):
                                continue
                            for i, j, k in voxels_junction:
                                if arr_label[i, j, k] == 0:
                                    arr_label[i, j, k] = val_aims_ss + \
                                        add_val['aims_top']

                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_label[i, j, k] = val_aims_ss + add_val[bucket_name]

    return vol_label


def generate_foldlabel_wide_junction(
        graph: aims.Graph) -> aims.Volume:
    """Converts an aims graph into skeleton and foldlabel volumes

    It should produce wide junctions as edges (junction, plidepassage)
    are written after vertices (aims_ss, aims_bottom).
    Thus, when voxels are present in both, junction voxels override
    simple surface and bottom voxels
    """

    vol_label = create_empty_volume_from_graph(graph)
    arr_label = np.asarray(vol_label)

    # Sorted in ascendent priority
    label = {'aims_other': 1,
             'aims_ss': 1000,
             'aims_bottom': 2000,
             'aims_junction': 3000,
             'aims_plidepassage': 4000}

    for vertex in graph.vertices():
        for bucket_name, value in {'aims_other': 100,
                                   'aims_ss': 60,
                                   'aims_bottom': 30}.items():
            bucket = vertex.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_label[i, j, k] = label[bucket_name]

    for edge in graph.edges():
        for bucket_name, value in {'aims_junction': 110}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_label[i, j, k] = label[bucket_name]

    for edge in graph.edges():
        for bucket_name, value in {'aims_plidepassage': 120}.items():
            bucket = edge.get(bucket_name)
            label[bucket_name] += 1
            if bucket is not None:
                voxels = np.array(bucket[0].keys())
                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_label[i, j, k] = label[bucket_name]

    return vol_label


def generate_foldlabel_from_graph(
        graph: aims.Graph,
        junction: str = _JUNCTION_DEFAULT) -> aims.Volume:
    """Generates foldlabel from graph"""
    if junction == 'wide':
        vol_label = generate_foldlabel_wide_junction(graph)
    else:
        vol_label = generate_foldlabel_thin_junction(graph)
    return vol_label


def generate_foldlabel_from_graph_file(graph_file: str,
                                       foldlabel_file: str,
                                       junction: str = _JUNCTION_DEFAULT):
    """Generates skeleton from graph file"""
    graph = aims.read(graph_file)
    vol_label = generate_foldlabel_from_graph(graph, junction)
    aims.write(vol_label, foldlabel_file)
