# /usr/bin/env python2.7 + brainvisa compliant env
# coding: utf-8
"""
The aim of this script is to create dataset from MRIs saved in a .pickle.
Several steps are required: normalization, crop and .pickle generation
"""
import os
import json
from load_data import fetch_data

side = 'L' # hemisphere 'L' or 'R'
bbox = ([105, 109, 23], [147, 171, 93]) # Bounding box defined thanks to
                                        # bbox_definition.py for S.T.s ter. asc.
                                        # ant. and post.

dir = '/neurospin/hcp/ANALYSIS/3T_morphologist/'
comp_dir = '/t1mri/default_acquisition/default_analysis/segmentation/' + side + 'skeleton_'
new_dir = '/neurospin/dico/deep_folding/data/' +  side + 'crops/'

for sub in os.listdir(dir): # go through all HCP subjects folder

    # Normalization and resampling of skeleton images
    dir_m = '/neurospin/dico/lguillon/skeleton/transfo_pre_process/natif_to_template_spm_' + sub +'.trm'
    dir_r = '/neurospin/hcp/ANALYSIS/3T_morphologist/' + sub + '/t1mri/default_acquisition/normalized_SPM_' + sub +'.nii'
    cmd_normalize = 'AimsResample -i ' + dir + sub + comp_dir + sub + '.nii.gz' + ' -o ' + new_dir + sub + '_normalized.nii.gz -m ' + dir_m + ' -r ' + dir_r
    os.system(cmd_normalize)

    # Crop of the images based on bouding box
    file = new_dir + sub + '_normalized.nii.gz'
    cmd_crop = 'AimsSubVolume -i ' + file + ' -o ' + file + ' -x 105 -y 109 -z 23 -X 147 -Y 171 -Z 93'
    os.system(cmd_crop)

# Creation of .pickle file
fetch_data(new_dir, save_dir='/neurospin/dico/deep_folding/data/', side=side)


input_dict = {'Bbox': bbox, 'side': side}
log_file = open(new_dir + 'logs.json', 'a+')
log_file.write(json.dumps(input_dict))
log_file.close()
