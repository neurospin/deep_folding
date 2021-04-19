# -*- coding: utf-8 -*-
# /usr/bin/env python2.7 + brainvisa compliant env
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

""" Getting bounding box from a list of sulci and specific hemisphere

The aim of this script is to output the bounding box of one or several sulci of
a specified hemisphere.

"""

from os.path import join
import json
from deep_folding.anatomist_tools.bounding_box import BoundingBoxMax


_SRC_DIR_DEFAULT = "/neurospin/dico/deep_folding_data/default/bbox/"


def load(sulci_list, side, talairach_box=False):
    """Function returning maximal bounding box of a given list of sulci

    Args:
        sulci_list: a list of sulci
        side: a string corresponding to the hemisphere, whether 'L' or 'R'
        talairach_box: a boolean whether using Talairach coordinates or voxels

    Returns:
        bbmin: an array of minimum coordinates of bounding box of given sulci
        bbmax: an array of maximum coordinates of bounding box of given sulci
    """
    
    list_bbmin, list_bbmax = [], []
    if talairach_box:
        rad = 'AIMS_Talairach'
    else :
        rad = 'voxel'

    for file in sulci_list:
        with open(join(_SRC_DIR_DEFAULT, side, file+'.json')) as json_file:
            sulcus = json.load(json_file)

            list_bbmin.append(sulcus['bbmin_'+rad])
            list_bbmax.append(sulcus['bbmax_'+rad])

    bbmin, bbmax = BoundingBoxMax.compute_max_box(list_bbmin, list_bbmax)

    return bbmin, bbmax


if __name__ =='__main__':
    bbmin, bbmax = load(['S.T.s.ter.asc.ant._left', 'S.T.s.ter.asc.test._left'],
                        'L')
    print("bbmin = ", bbmin)
    print("bbmax = ", bbmax)