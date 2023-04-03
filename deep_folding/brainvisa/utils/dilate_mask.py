# -*- coding: utf-8 -*-
# /usr/bin/env python3.6 + brainvisa compliant env
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
# knowledge of the CeCILL license version 2 and that you accept its
# terms.""" Dilating a given mask"""from soma import aims


import numpy as np
from soma import aims
from soma.aimsalgo import MorphoGreyLevel_S16

# import anatomist.api as anatomist
# from soma.qt_gui.qt_backend import Qt

_AIMS_BINARY_ONE = 32767


def dilate(mask, radius=10.):
    """
    """
    # Creates volume
    # hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
    # vol = aims.Volume(hdr['volume_dimension'], dtype='S16')
    # vol.copyHeaderFrom(hdr)
    arr = np.asarray(mask)
    # Binarization of mask
    arr[arr < 1] = 0
    arr[arr >= 1] = _AIMS_BINARY_ONE
    # Dilates initial volume of 10 mm
    morpho = MorphoGreyLevel_S16()
    dilate = morpho.doDilation(mask, radius)
    arr_dilate = np.asarray(dilate)
    arr_dilate[arr_dilate >= 1] = 1
    return dilate


def main():
    mask = aims.read(
        '/neurospin/dico/data/deep_folding/current/mask/2mm/R/'
        'paracingular._right.nii.gz')
    mask_dilated = dilate(mask)
    aims.write(mask_dilated, '/tmp/mask_dil.nii.gz')


if __name__ == '__main__':
    main()
