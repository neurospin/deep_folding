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

"Utilities to generate distance maps from skeleton files"

import os
import numpy as np
import tempfile
from soma import aims
from soma import aimsalgo

from deep_folding.brainvisa.utils.padding import padd
from deep_folding.config.logs import set_file_logger

# Defines logger
log = set_file_logger(__file__)


def generate_distmap_from_skeleton_file(skeleton_file: str,
                                        distmap_file: str):
    """Generates distmap from skeleton file.
    Distmaps files are padded to avoid 0-background close to skeleton voxels
    when going to ICBMc referential.
    /!\\ skeleton files have various dimensions"""
    # temporary directory
    temp_dir = tempfile.mkdtemp()

    # 200 voxels will be added in each dimension
    dim_padd = (200, 200, 200)
    nb_vox = dim_padd[0] / 2

    # loading of skeleton from which distmap will be generated
    orig_skel = aims.read(skeleton_file)
    orig_skel_arr = np.asarray(orig_skel)

    # initial dimensions of the skeleton
    dim_ini = orig_skel_arr.shape
    dim = tuple(map(sum, zip(dim_ini, dim_padd)))

    # creation of an empty volume of new dimension: original dim +
    # padded_voxels
    vol_ref = aims.Volume(dim, dtype='S16')
    vol_ref.copyHeaderFrom(orig_skel.header())
    in_voxel_size = orig_skel.header()['voxel_size']

    # definition of translation (half added voxels -> origin is in top left
    # corner)
    translation = (
        nb_vox * in_voxel_size[0],
        nb_vox * in_voxel_size[1],
        nb_vox * in_voxel_size[2])
    distmap_to_padded_distmap = aims.AffineTransformation3d()
    distmap_to_padded_distmap.setTranslation(translation)

    list_transfo = []
    # combination of translation with header existing transformations
    for transfo in orig_skel.header()['transformations']:
        new_transfo = np.asarray(transfo) * \
            np.asarray(distmap_to_padded_distmap.inverse().toVector())
        list_transfo.append(new_transfo)
    # writing of new transformations that take into account translation
    # due to padding
    vol_ref.header()['transformations'] = list_transfo

    # Increasing of dimensions of skeletons in order to avoid 0-background at
    # resampling
    temp_vol = padd(orig_skel_arr, dim, fill_value=0)
    temp_file = f"{temp_dir}/skel_new_dim.nii.gz"
    np.asarray(vol_ref)[:] = temp_vol
    aims.write(vol_ref, temp_file)

    # Generation of distmap from padded skeletons
    cmd_distMap = 'AimsChamferDistanceMap' + \
        ' -i ' + temp_file + \
        ' -o ' + distmap_file + \
        ' -s OUTSIDE'
    log.debug(cmd_distMap)
    os.system(cmd_distMap)


def generate_distmap_from_resampled_skeleton(skeleton_file: str,
                                             distmap_file: str):
    """Generates distmap from resampled skeleton file."""

    # Generation of distmap from padded skeletons
    cmd_distMap = 'AimsChamferDistanceMap' + \
        ' -i ' + skeleton_file + \
        ' -o ' + distmap_file + \
        ' -s OUTSIDE'
    log.debug(cmd_distMap)
    os.system(cmd_distMap)
