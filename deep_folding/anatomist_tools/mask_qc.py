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

"""Quality check of generated masks

"""

from soma import aims
from soma import aimsalgo as ago


def check(subject):
    """
    """
    # Loading of subject graph
    graph_dir = f"/home/lg261972/Documents/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist/{subject}/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/R{subject}_default_session_auto.arg"
    graph = aims.read(graph_dir)

    # Loading of subject skeleton
    skeleton_dir = f"/home/lg261972/Documents/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist/{subject}/t1mri/default_acquisition/default_analysis/segmentation/Rskeleton_{subject}.nii.gz"
    skeleton = aims.read(skeleton_dir)

    masked_resampled = aims.Volume(skeleton.header()['volume_dimension'][:3], dtype=skeleton.__array__().dtype)
    masked_resampled.header()['voxel_size'] = skeleton.header()['voxel_size'][:3]

    g_to_icbm = aims.GraphManip.getICBM2009cTemplateTransform(graph)

    g_to_rw = g_to_icbm.inverse()

    mask = aims.read(f"/nfs/neurospin/dico/data/deep_folding/new/mask/2mm/R/S.C._right.nii.gz")

    resampler = ago.ResamplerFactory(mask).getResampler(0)
    resampler.setDefaultValue(0)
    resampler.setRef(mask)
    resampler.resample(mask, g_to_rw, 0, masked_resampled)

    aims.write(masked_resampled, f"/nfs/neurospin/dico/data/deep_folding/new/QC_skeleton/mask_resampled_{subject}.nii.gz")


check('585862')
