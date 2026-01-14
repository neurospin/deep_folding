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

"Generates spam"

import json
import numpy as np
from soma import aims
from deep_folding import config

root = config.config().get_champollion_data_root_dir()
json_path = f"{root}/sulci_regions_gridsearch.json"
mask_dir = f"{root}/mask/2mm"
output_path = f"{root}/mask/2mm/regions"
sides = ['R', 'L']
regions = [
    "F.I.P.", "S.C.-sylv.", "S.C.-S.Pe.C.", "S.C.-S.Po.C.",
    "S.Pe.C.", "S.Po.C.", "CINGULATE.", "S.F.int.-F.C.M.ant.",
    "S.F.inf.-BROCA-S.Pe.C.inf.", "S.T.s.", "Sc.Cal.-S.Li.",
    "F.C.M.post.-S.p.C.", "S.T.i.-S.O.T.lat.",
    "OCCIPITAL", "S.F.inter.-S.F.sup.",
    "S.F.median-S.F.pol.tr.-S.F.sup.", "S.Or.",
    "S.Or.-S.Olf.", "F.P.O.-S.Cu.-Sc.Cal.",
    "S.s.P.-S.Pa.int.", "S.T.s.br.",
    "Lobule_parietal_sup.", "S.F.marginal-S.F.inf.ant.",
    "F.Coll.-S.Rh.", "S.T.i.-S.T.s.-S.T.pol.",
    "F.C.L.p.-subsc.-F.C.L.a.-INSULA.", "S.F.int.-S.R.",
    "fronto-parietal_medial_face.",
    "S.T.s.-S.GSM.", "F.C.L.p.-S.GSM."
    ]

for side in sides:
    for region_name in regions:
        print(f"REGION = {region_name} -- SIDE = {side}")
        full_side = "_right" if side == 'R' else "_left"

        # Reads json file containing all predfined sulcal regions
        # and gets corresponding sulci
        with open(json_path, 'r') as file:
            brain_regions = json.load(file)
        try:
            full_name = region_name + full_side
            sulci_list = list(brain_regions['brain'][full_name].keys())
        except ValueError:
            print(
                f"The given region {region_name} "
                f"is not in the dictionary at {json_path}")


        # Reads the list of masks as a list of aims volumes
        list_masks = []
        hdr = aims.StandardReferentials.icbm2009cTemplateHeader()


        for sulcus in sulci_list:
            mask_file = f"{mask_dir}/{side}/{sulcus}.nii.gz"
            print(f"mask file: {mask_file}")
            list_masks.append(aims.read(mask_file))


        # Computes the mask being a combination of all masks
        mask_result = aims.Volume(list_masks[0].shape, dtype='S16')
        mask_result.copyHeaderFrom(hdr)
        mask_result.header()['voxel_size'] = list_masks[0].header()['voxel_size']
        mask_result_arr = np.asarray(mask_result)
        mask_result_arr[:] = (np.asarray(list_masks[0])).copy()
        
        arr_result = np.asarray(mask_result)
        print(f"first sulcus: {sulci_list[0]}")
        for k, mask in enumerate(list_masks[1:]):
            print(f"sulcus {sulci_list[k+1]}")
            arr = np.asarray(mask)
            arr_result += arr


        # Writes the resulting mask
        output_file = f"{output_path}/{side}/{region_name}{full_side}.nii.gz"
        aims.write(mask_result, output_file)
