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

The aim of this script is to create dataset of cropped skeletons from MRIs saved in a .pickle.
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


######################################################################
# Global variables (that the user can change)
######################################################################

side = 'L' # hemisphere 'L' or 'R'

# Bounding box defined thanks to
# bbox_definition.py for S.T.s ter. asc.
# ant. and post. 
bbox = ([105, 109, 23], [147, 171, 93])                        


# Input directories
# --------------

# Directory that contains the transformation file
# from native to MNI through SPM
# These files have been created with spm_skeleton
# example: dir_input_transform = '/neurospin/dico/lguillon/skeleton/transfo_pre_process' 
dir_input_transform = '/neurospin/dico/deep_folding/data/transfo_pre_process'

# Input directory contaning the morphologist analysis of the HCP database
dir_input_MRI = '/neurospin/hcp/ANALYSIS/3T_morphologist'

# Output directory
# --------------
dir_output_base = '/neurospin/dico/deep_folding/data'


######################################################################
# Global variables (that the user will probably not change)
######################################################################

# Take the coordinates of the bounding box
xmin, ymin, zmin = str(bbox[0][0]), str(bbox[0][1]), str(bbox[0][2])
xmax, ymax, zmax = str(bbox[1][0]), str(bbox[1][1]), str(bbox[1][2])

# Define the subdirectory in which the cropped skeletons will be saved
subdir = side + 'crops'
dir_output = join(dir_output_base, side + 'crops')


######################################################################
# Main function
######################################################################


def main():
  """Main loop to create pickle files
  
  The programm loops over all the subjects from the input directory.
  """

  for sub in os.listdir(dir_input_MRI)[:2]: # go through all HCP subjects folder

      # Creating transformation file name
      file_transform_basename = 'natif_to_template_spm_' + sub + '.trm' 
      file_transform = join(dir_input_transform, file_transform_basename) 
      
      # Creating normalized SPM file name
      file_SPM_basename = 'normalized_SPM_' + sub +'.nii'
      file_SPM = join(dir_input_MRI, sub, 't1mri/default_acquisition', file_SPM_basename)
      
      # Creating skeleton file name
      file_skeleton_basename = side + 'skeleton_' + sub + '.nii.gz'
      file_skeleton = join(dir_input_MRI, sub, 't1mri/default_acquisition/default_analysis/segmentation', file_skeleton_basename)
      
      # Creating output file name
      file_output_basename = sub + '_normalized.nii.gz'
      file_output = join(dir_output, file_output_basename)

      # Normalization and resampling of skeleton images
      cmd_normalize = 'AimsResample' + ' -i ' + file_skeleton + ' -o ' + file_output + ' -m ' + file_transform + ' -r ' + file_SPM
      os.system(cmd_normalize)

      # Crop of the images based on bounding box
      cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + ' -X '+ xmax + ' -Y ' + ymax + ' -Z ' + zmax
      cmd_crop = 'AimsSubVolume' + ' -i ' + file_output + ' -o ' + file_output + cmd_bounding_box
      os.system(cmd_crop)

  # Creation of global .pickle file
  fetch_data(dir_output, save_dir=dir_output_base, side=side)

  # Log information
  input_dict = {'Bbox': bbox, 'side': side}
  log_file_name = join(dir_output, 'logs.json')
  log_file = open(log_file_name, 'a+')
  log_file.write(json.dumps(input_dict))
  log_file.close()
  
  
######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    main()
