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

"Utilities on graphs"

from soma import aims
from deep_folding.config.logs import set_file_logger

# Defines logger
log = set_file_logger(__file__)


def create_empty_volume_from_graph(graph: aims.Graph, dimensions: list = None) -> aims.Volume:
    """Creates empty volume with graph header"""

    voxel_size = graph['voxel_size'][:3]
    if dimensions is None:
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
