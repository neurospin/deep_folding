import numpy as np
import os
import pandas as pd
from soma import aims
import warnings
from tqdm import tqdm


"""
This file masks the resampled_foldlabel files generated using resample_files.
For now, foldlabel needs to be run twice: once for resampled_foldlabel generation,
and once for cropping AFTER applying applying the following mask.
"""

def nearest_nonzero_idx(a,x,y,z):
    tmp = a[x,y,z]
    a[x,y,z] = 0
    d,e,f = np.nonzero(a)
    a[x,y,z] = tmp
    min_idx = ((d - x)**2 + (e - y)**2 + (f - z)**2).argmin()
    return(d[min_idx], e[min_idx], f[min_idx])

dataset='UkBioBank'
#dataset='ACCpatterns'
root = '/neurospin/dico/data/deep_folding/current/datasets/'
#root = '/volatile/jl277509/data/' # but I copy only the crops locally..
resolution, res = "1.5mm", 1.5
side='R'
resume=False

vx_tolerance=30

old_dir = f'{root}{dataset}/foldlabels/{resolution}_tmp/'
new_dir = f'{root}{dataset}/foldlabels/{resolution}/'
old_foldlabel_dir = os.path.join(old_dir,side)
new_foldlabel_dir = os.path.join(new_dir,side)
skels_dir = f'{root}{dataset}/skeletons/{resolution}/{side}/'
subjects = os.listdir(skels_dir)
skel_subjects = [sub[20:-7] for sub in subjects if sub[-1]!='f']
print(f'number of subjects detected in {dataset}: {len(skel_subjects)}')
print(f'first skel subjects: {skel_subjects[:5]}')

#check existence of foldlabels
if not resume:
    assert os.path.isdir(new_dir), "Compute resampled foldlabels using deep_folding before masking."
    #move foldlabels to tmp folder 
    os.rename(new_dir, old_dir)
    os.makedirs(new_foldlabel_dir)

if resume:
    # list already computed subjects
    subjects_already_masked = os.listdir(new_foldlabel_dir)
    subjects_already_masked = [elem[21:-7] for elem in subjects_already_masked if elem[-1]!='f']
    skel_subjects = [elem for elem in skel_subjects if elem not in subjects_already_masked]
    print(f'{len(subjects_already_masked)} subjects already processed, resuming')

for i, subject in enumerate(tqdm(skel_subjects)):

    skel = aims.read(os.path.join(skels_dir,f'{side}resampled_skeleton_{subject}.nii.gz'))
    old_foldlabel = aims.read(os.path.join(old_foldlabel_dir,f'{side}resampled_foldlabel_{subject}.nii.gz'))
    skel_np = skel.np
    old_foldlabel_np = old_foldlabel.np

    foldlabel = old_foldlabel_np.copy()
    # first mask skeleton using foldlabel because sometimes 1vx is added during skeletonization...
    foldlabel[skel_np==0]=0
    f = foldlabel!=0
    s = skel_np!=0
    diff_fs = np.sum(f!=s)
    assert (diff_fs<=vx_tolerance), f"subject {subject} has incompatible foldlabel and skeleton. {np.sum(s)} vx in skeleton, {np.sum(f)} vx in foldlabel"
    if diff_fs!=0:
        warnings.warn(f"subject {subject} has incompatible foldlabel and skeleton. {np.sum(s)} vx in skeleton, {np.sum(f)} vx in foldlabel")
        idxs = np.where(f!=s)
        print(idxs)
        for i in range(diff_fs):
            x,y,z = idxs[0][i], idxs[1][i], idxs[2][i]
            d,e,f = nearest_nonzero_idx(foldlabel[:,:,:,0],x,y,z)
            foldlabel[x,y,z,0]=foldlabel[d,e,f,0]
            print(f'foldlabel has a 0 at index {x,y,z}, nearest nonzero at index {d,e,f}, value {foldlabel[d,e,f,0]}')
    f = foldlabel!=0
    assert np.sum(f!=s)==0, f'subject {subject} has incompatible foldlabel and skeleton AFTER CORRECTION. {np.sum(s)} vx in skeleton, {np.sum(f)} vx in foldlabel'
    vol = aims.Volume(foldlabel)
    vol.header()['voxel_size'] = [res, res, res]
    aims.write(vol, os.path.join(new_foldlabel_dir,f'{side}resampled_foldlabel_{subject}.nii.gz'))