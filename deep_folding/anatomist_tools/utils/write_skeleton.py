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

"""Write skeletons from graph files

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python write_skeleton.py


"""

from soma import aims
import numpy as np
import dico_toolbox as dtx

graph_filename = '/mnt/n4hhcp/hcp/ANALYSIS/3T_morphologist/299760/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/R299760_default_session_auto.arg'

graph = aims.read(graph_filename)
voxel_size_in = graph['voxel_size'][:3]
voxel_size_out = graph['voxel_size'][:3]
dimensions = graph['boundingbox_max']

vol = aims.Volume(dimensions, dtype='S16')
vol.header()['voxel_size'] = voxel_size_out
vol.header()['transformations'] = graph['transformations']
vol.header()['referentials'] = graph['referentials']
arr = np.asarray(vol)

for vertex in graph.vertices():
    for bucket_name, value in {'aims_other':100, 'aims_ss':60, 'aims_bottom':30}.items():
        bucket = vertex.get(bucket_name)
        if bucket is not None:
            voxels_real = np.asarray([(np.array(voxel) * voxel_size_in) for voxel in bucket[0].keys()])
            if voxels_real.shape == (0,):
                continue
            voxels = np.round(np.array(voxels_real) / voxel_size_out[:3]).astype(int)
            for i,j,k in voxels:
                arr[i,j,k] = value

aims.write(vol, '/tmp/skel.nii.gz')
