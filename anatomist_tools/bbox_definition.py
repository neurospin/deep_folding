"""
Script that outputs bounding box for a given sulci based on a manually
labeled dataset.
Bounding box corresponds to the biggest box encountered in the manually
labeled subjects.
"""

from soma import aims
import os
import numpy as np

root_dir = "/neurospin/lnao/PClean/database_learnclean/all/"

subjects = []

for filename in os.listdir(root_dir):
    directory = os.path.join(root_dir, filename)
    if os.path.isdir(directory):
        if filename != 'ra':
            subjects.append(filename)

# Bouding box in Talairach space
atts = {'subject': 'ammon', 'side': 'L'}
list_bbmin = []
list_bbmax = []

for sub in subjects:
    if sub != 'eros':
        sulci_pattern = root_dir+'%(subject)s/t1mri/t1/default_analysis/folds/3.3/base2018_manual/%(side)s%(subject)s_base2018_manual.arg'
        sulcus = 'S.T.s.ter.asc.ant._left'

        atts['subject'] = sub
        graph = aims.read(sulci_pattern % atts)

        tal_transfo = aims.GraphManip.talairach(graph)
        voxel_size = graph['voxel_size'][:3]
        bbox_min = None
        bbox_max = None

        for vertex in graph.vertices():
            vname = vertex.get('name')
            if vname != sulcus:
                continue
            for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
                bucket = vertex.get(bucket_name)

                voxels = np.asarray(
                    [tal_transfo.transform(np.array(voxel) * voxel_size)
                     for voxel in bucket[0].keys()])

                if voxels.shape == (0, ):
                    continue
                bbox_min = np.min(np.vstack(
                    ([bbox_min] if bbox_min is not None else [])
                    + [voxels]), axis=0)
                bbox_max = np.max(np.vstack(
                    ([bbox_max] if bbox_max is not None else [])
                    + [voxels]), axis=0)

        print('bounding box min:', bbox_min)
        print('bounding box max:', bbox_max)

        list_bbmin.append([bbox_min[0], bbox_min[1], bbox_min[2]])
        list_bbmax.append([bbox_max[0], bbox_max[1], bbox_max[2]])

ave_bbmin = np.array([min([val[0] for k, val in enumerate(list_bbmin)]),
                      min([val[1] for k, val in enumerate(list_bbmin)]),
                      min([val[2] for k, val in enumerate(list_bbmin)])])

ave_bbmax = np.array([max([val[0] for k, val in enumerate(list_bbmax)]),
                      max([val[1] for k, val in enumerate(list_bbmax)]),
                      max([val[2] for k, val in enumerate(list_bbmax)])])


tal_to_mni = aims.read(aims.carto.Paths.findResourceFile('transformation/talairach_TO_spm_template_novoxels.trm'))

# To go back to HCP space
# vol = aims.read('/neurospin/hcp/ANALYSIS/3T_morphologist/100307/t1mri/default_acquisition/100307.nii.gz')

# Space of Jeff's original crops
#vol = aims.read('/neurospin/hcp/ANALYSIS/3T_morphologist/100206/t1mri/default_acquisition/normalized_SPM_100206.nii')
vol = aims.read('Rskeleton_159946_normalized_crop.nii.gz')

template_mni = aims.read('/neurospin/dico/lguillon/MNI152_T1_1mm.nii.gz')
mni_to_template = aims.AffineTransformation3d(template_mni.header()['transformations'][0])

tal_to_template = mni_to_template *tal_to_mni
#vol_to_mni = vol.header()['transformations'][-1] # ici c'est la derniere transfo
vol_to_mni = aims.AffineTransformation3d(vol.header()['transformations'][-1])
tal_to_vol = vol_to_mni.inverse() * tal_to_mni

# Application of the transformation to bbox
ave_bbmin = tal_to_vol.transform(ave_bbmin)
ave_bbmax = tal_to_vol.transform(ave_bbmax)

# To go back from mms to voxels
vs = vol.header()['voxel_size'][:3]

vox_bbmin = np.round(np.array(ave_bbmin) / vs).astype(int)
vox_bbmax = np.round(np.array(ave_bbmax) / vs).astype(int)

print(vox_bbmin, vox_bbmax)
