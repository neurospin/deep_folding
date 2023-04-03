#!/usr/bin/env python
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

import numpy as np
from soma import aims

from deep_folding.config.logs import set_file_logger

# Defines logger
log = set_file_logger(__file__)


def generate_ref_volume_ICBM2009c(out_voxel_size: tuple) -> aims.Volume:
    """Defines MNI 2009 reference aims volume with output voxel size

    Args:
        output_voxel_size: tuple
            Output voxel size (default: None, no resampling)

    Returns:
        vol: volume (aims.Volume_S16) filled with 0 in MNI2009 referential
            and with requested voxel_size

    """
    hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
    voxel_size = np.concatenate((out_voxel_size, [1]))
    resampling_ratio = np.array(hdr['voxel_size']) / voxel_size

    orig_dim = hdr['volume_dimension']
    new_dim = list((resampling_ratio * orig_dim).astype(int))

    vol = aims.Volume(new_dim, dtype='S16')
    vol.copyHeaderFrom(hdr)
    vol.header()['voxel_size'] = voxel_size

    return vol


def ICBM2009c_to_aims_talairach(point_ICBM2009c: np.array) -> np.array:
    """Transforms coordinates from ICBM2009c to AIMS talairach referential"""

    g_icbm_template_to_talairach = \
        aims.StandardReferentials.talairachToICBM2009cTemplate().inverse()
    point_tal = g_icbm_template_to_talairach.transform(point_ICBM2009c)

    return np.asarray(point_tal)
