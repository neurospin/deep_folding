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

"Utilities to generate trimmed skeletons from graph and skeleton with hull"

import numpy as np
from soma import aims
from soma.aimsalgo.sulci import trim_extremity

from deep_folding.config.logs import set_file_logger

# Defines logger
log = set_file_logger(__file__)


def is_volume_binary(arr):
    """checks if arr is binary"""
    arr_values = np.array([0, 1])
    return np.isin(arr, arr_values).all()


def generate_extremities_from_graph(graph: aims.Graph,
                                    skeleton_with_hull: aims.Volume) -> aims.Volume:
    """Generete extremities (dilated lateral edges).
    Returns volume"""
    trm = aims.GraphManip.getICBM2009cTemplateTransform(graph)
    trm = trm.np
    scale = (trm[0, 0] * trm[1, 1] * trm[2, 2]) ** (1 / 3)
    log.debug(f"transform = {trm}")
    log.debug(f"scale = {scale}")
    tminss = 3. / scale
    ss, trimmed = trim_extremity.trim_extremities(skeleton_with_hull,
                                                  graph,
                                                  tminss,
                                                  junc_dilation=1)
    ss.np[ss.np < 32500] = 1
    ss.np[ss.np >= 32500] = 0

    extremities = ss - trimmed

    log.info(f"Non zero voxels ratio before/after trimming : "
             f"{np.sum(ss.np!=0)} / {np.sum(trimmed.np!=0)}")

    return extremities


def generate_extremities_from_graph_file(graph_file: str,
                                         skeleton_with_hull_file: str,
                                         extremity_file: str):
    """Generates extremities from graph file and skeleton with hull.
    
    The extremities are only the inflated lateral edges of branches"""

    graph = aims.read(graph_file)
    skeleton_with_hull = aims.read(skeleton_with_hull_file)
    vol_extremities = generate_extremities_from_graph(
                                                    graph,
                                                    skeleton_with_hull)
    if not is_volume_binary(vol_extremities.np):
        raise ValueError(f"{skeleton_with_hull_file} has non-binary values")
    aims.write(vol_extremities, extremity_file)
