import numpy as np
import os
from soma import aims, aimsalgo
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_map

from deep_folding.brainvisa.utils.parallel import define_njobs

"""
Computes distbottom nifti and npy files from cropped skeleton.
For now, this code works on crops, not on the whole brain.
"""

directory = f'/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/S.C.-S.Pe.C./mask/'
side = 'R'

crops_dirs = directory+side+'crops/'
skel_subjects = pd.read_csv(directory+side+'skeleton_subject.csv') # NB: skeleton_subject needs to be in consistent order with Rskeleton.npy

os.makedirs(f'{directory}{side}distbottom', exist_ok=True)

def make_distbottom(subject):
    vol = aims.read(crops_dirs+subject+'_cropped_skeleton.nii.gz')
    outside = 0
    other_outside = 11
    bottom_val = 30
    ss_val = 60

    # change all other values to ss_val
    vol_tmp = aims.Volume(vol)
    vol[vol.np != ss_val] = ss_val
    vol[vol_tmp.np == bottom_val] = bottom_val
    vol[vol_tmp.np == outside] = outside
    vol[vol_tmp.np == other_outside] = outside

    # propagation dans ss_val, avec outside non-atteignable
    aimsalgo.AimsDistanceFrontPropagation(vol, ss_val, outside, 3, 3, 3, 50, False)
    aims.write(vol, directory+side+f'distbottom/{subject}_cropped_distbottom.nii.gz')

p_map(
    make_distbottom,
    skel_subjects.Subject,
    num_cpus=define_njobs())

distbottom_list = []
for i, subject in enumerate(tqdm(skel_subjects.Subject)):
    vol = aims.read(directory+side+f'distbottom/{subject}_cropped_distbottom.nii.gz')
    distbottom_list.append(vol.np)

# save al distbottoms in numpy array
arr = np.stack(distbottom_list)
print(arr.shape)
np.save(directory+side+'distbottom.npy', arr)
