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

"""Create numpy arrays from folders containing skeletons, distance maps...

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python augmentations_tools.py


"""


"""Aim: create a numpy array of all nii.gz files from a given folder
     create a numpy array with all corresponding id_subjects

     /!\ id must correspond to file"""


import os
import glob

import numpy as np
import re
from soma import aims


def generate_np_array(src_dir):
    """
    """
    list_arr_id = []
    list_arr_file = []
    expr = '^.distmap_generated_([0-9a-zA-Z]*).nii.gz$'
    side = 'R'

    if os.path.isdir(src_dir):
        list_all_subjects = \
            [re.search(expr, os.path.basename(dI))[1]
             for dI in glob.glob(f"{src_dir}/*.nii.gz")]

    else:
        raise NotADirectoryError(
            f"{self.src_dir} doesn't exist or is not a directory")

    for sub in list_all_subjects:
        src_file = os.path.join(src_dir, f"{side}distmap_generated_{sub}.nii.gz")
        file = aims.read(src_file)
        arr_file = np.asarray(file)
        list_arr_file.append(arr_file)
        list_arr_id.append(sub)

    list_arr_id = np.array(list_arr_id)
    list_arr_file = np.array(list_arr_file)
    np.save(os.path.join(src_dir, 'sub_id.npy'), list_arr_id)
    np.save(os.path.join(src_dir, 'data.npy'), list_arr_file)


def main():
    generate_np_array('/neurospin/dico/data/deep_folding/current/datasets/' \
                      'hcp/distmaps/2mm/R')


if __name__ == '__main__':
    main()
