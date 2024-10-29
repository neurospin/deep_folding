import json
import numpy as np
from soma import aims

json_path = "/neurospin/dico/data/deep_folding/current/sulci_regions_gridsearch.json"
mask_dir = f"/neurospin/dico/data/deep_folding/current/mask/2mm"
output_path = "/neurospin/dico/data/deep_folding/current/mask/2mm/regions"
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


        # Writes the resulting mask
        output_file = f"{output_path}/{side}/{region_name}{full_side}.nii.gz"
        aims.write(mask_result, output_file)