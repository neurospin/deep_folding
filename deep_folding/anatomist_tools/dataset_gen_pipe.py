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

"""Creating pickle file from T1 MRI datas

The aim of this script is to create dataset of cropped skeletons from MRIs
saved in a .pickle file.
Several steps are required: normalization, crop and .pickle generation

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python dataset_gen_pipe.py

  Alternatively, you can launch the script in the interactive terminal ipython:
  >>> %run dataset_gen_pipe.py

"""

######################################################################
# Imports and global variables definitions
######################################################################

import os
from os.path import join
import json

from load_data import fetch_data

_ALL_SUBJECTS = -1

_SIDE_DEFAULT = 'L' # hemisphere 'L' or 'R'

# Bounding box defined thanks to
# bbox_definition.py for S.T.s ter. asc.
# ant. and post. 
bbox = ( ([112, 110, 24], [147, 152, 78]) if side=='L' # bbox for left side: 'L'
         else ([8, 95, 23], [43, 146, 85]) ) # bbox for right side: 'R'  

# Input directories
# -----------------

# Directory that contains the transformation file
# from native to MNI through SPM
# These files have been created with spm_skeleton
_TRANSFORM_DIR_DEFAULT = '/neurospin/dico/deep_folding_data/default/transform'

# Input directory contaning the morphologist analysis of the HCP database
_SRC_DIR_DEFAULT = '/neurospin/hcp/ANALYSIS/3T_morphologist'

# Output (target) directory
# -------------------------
_TGT_DIR_DEFAULT = '/neurospin/dico/deep_folding_data/default'


######################################################################
# Global variables (that the user will probably not change)
######################################################################

# Take the coordinates of the bounding box
xmin, ymin, zmin = str(bbox[0][0]), str(bbox[0][1]), str(bbox[0][2])
xmax, ymax, zmax = str(bbox[1][0]), str(bbox[1][1]), str(bbox[1][2])

# Define the subdirectory in which the cropped skeletons will be saved
subdir = side + 'crops'
dir_output = join(tgt_dir, side + 'crops')


######################################################################
# Main function
######################################################################


def dataset_gen_pipe(transform_dir=_TRANSFORM_DIR_DEFAULT,
                     src_dir=_SRC_DIR_DEFAULT,
                     tgt_dir=_TGT_DIR_DEFAULT,
                     number_subjects=_ALL_SUBJECTS
                     ):
  """Main loop to create pickle files
  
  The programm loops over all the subjects from the input (source) directory.
  """

  if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

  for sub in os.listdir(src_dir): # go through all HCP subjects folder

      # Transformation file name
      file_transform_basename = 'natif_to_template_spm_' + sub + '.trm' 
      file_transform = join(transform_dir, file_transform_basename)
      
      # Normalized SPM file name
      file_SPM_basename = 'normalized_SPM_' + sub +'.nii'
      file_SPM = join(src_dir, sub, 't1mri/default_acquisition', file_SPM_basename)
      
      # Skeleton file name
      file_skeleton_basename = side + 'skeleton_' + sub + '.nii.gz'
      file_skeleton = join(src_dir, sub,
                           't1mri/default_acquisition/'
                           'default_analysis/segmentation',
                           file_skeleton_basename)
      
      # Creating output file name
      file_output_basename = sub + '_normalized.nii.gz'
      file_output = join(tgt_dir, file_output_basename)

      # Normalization and resampling of skeleton images
      cmd_normalize = 'AimsResample' + \
                      ' -i ' + file_skeleton + \
                      ' -o ' + file_output + \
                      ' -m ' + file_transform + \
                      ' -r ' + file_SPM
      os.system(cmd_normalize)

      # Crop of the images based on bounding box
      cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + \
                         ' -X '+ xmax + ' -Y ' + ymax + ' -Z ' + zmax
      cmd_crop = 'AimsSubVolume' + \
                 ' -i ' + file_output + \
                 ' -o ' + file_output + cmd_bounding_box
      os.system(cmd_crop)

  # Creation of .pickle file for all subjects
  fetch_data(tgt_dir, save_dir=dir_output_base, side=side)

  # Log information
  input_dict = {'Bbox': bbox, 'side': side}
  log_file_name = join(tgt_dir, 'logs.json')
  log_file = open(log_file_name, 'a+')
  log_file.write(json.dumps(input_dict))
  log_file.close()
  
  
######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
