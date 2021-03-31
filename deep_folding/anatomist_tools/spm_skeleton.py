# /usr/bin/env python2.7 + brainvisa compliant env
# coding: utf-8
"""
The aim of this script is to compute transformation files
"""
from __future__ import division
from soma import aims
import os

def calculate_transforms(subject_id):
    """
    Function that calculate transformation file of a given subject. This
    transformation enables to go from native space (= MRI subject space) to
    normalized SPM space.
    IN: subject_id: int or str, id of subject of whom transformation file is
                    computed
    OUT: no output, just saving of transformation file
    """
    # Directory where subjects are stored
    root_dir = "/neurospin/hcp/ANALYSIS/3T_morphologist/" + str(subject_id) + "/t1mri/default_acquisition/"
    # Directory where to store transformation files
    saved_folder = "/neurospin/dico/lguillon/skeleton/transfo_pre_process/"
    # Transformation file that goes from native space to MNI space
    natif_to_mni = aims.read(root_dir + 'registration/RawT1-'+subject_id+'_default_acquisition_TO_Talairach-MNI.trm')

    # Fetching of template's transformation
    template = aims.read(root_dir + 'normalized_SPM_'+subject_id+'.nii')
    mni_to_template = aims.AffineTransformation3d(template.header()['transformations'][0]).inverse()
    #print(template.header()['transformations'][0])

    # Combination of transformations
    natif_to_template_mni = mni_to_template * natif_to_mni
    #print(natif_to_template_mni)

    # Saving of transformation files
    aims.write(natif_to_template_mni, saved_folder + 'natif_to_template_spm_'+subject_id+'.trm')


root_dir_1 = "/neurospin/hcp/ANALYSIS/3T_morphologist/"
root_dir_2 = "/t1mri/default_acquisition/"
for subject in os.listdir(root_dir_1):
    #print(subject)
    #print(root_dir_1+str(subject)+root_dir_2)
    calculate_transforms(subject)
