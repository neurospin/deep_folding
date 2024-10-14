"""Uses the pipeline on multiple regions and datasets,
for both sides and input type."""

import os
import json
import subprocess

"""
regions = ["S.C.-sylv.", "S.C.-S.Pe.C.", "S.C.-S.Po.C.",\
            "S.Pe.C.", "S.Po.C.", "CINGULATE.", "S.F.int.-F.C.M.ant.",\
            "S.F.inf.-BROCA-S.Pe.C.inf.", "S.T.s.", "Sc.Cal.-S.Li.",\
            "F.C.M.post.-S.p.C.", "S.T.i.-S.O.T.lat.",\
            "OCCIPITAL", "F.I.P.", "S.F.inter.-S.F.sup.",\
            "S.F.median-S.F.pol.tr.-S.F.sup.", "S.Or.",\
            "S.Or.-S.Olf.", "F.P.O.-S.Cu.-Sc.Cal.",\
            "S.s.P.-S.Pa.int.", "S.T.s.br.",\
            "Lobule_parietal_sup.", "S.F.marginal-S.F.inf.ant.",\
            "F.Coll.-S.Rh.", "S.T.i.-S.T.s.-S.T.pol.",\
            "F.C.L.p.-subsc.-F.C.L.a.-INSULA.", "S.F.int.-S.R.",\
            "fronto-parietal_medial_face."\
            ]
regions = ["F.I.P.", "S.T.s.-S.GSM.", "F.C.L.p.-S.GSM."]
regions = ["F.I.P."]
regions = ["OCCIPITAL"]
regions = ["S.T.s.-S.GSM.", "F.C.L.p.-S.GSM."]
datasets = ["hcp", "UkBioBank"]
datasets = ["synesthetes"]
datasets = ["candi", "cnp", "bsnip1", "schizconnect-vip-prague"]
datasets = ["PreCatatoes"]
"""

path_dataset_root = "/neurospin/dico/data/deep_folding/current/datasets"
datasets = ["UkBioBank40"]
"""
regions = ["S.C.-sylv.", "S.C.-S.Pe.C.", "S.C.-S.Po.C.",
           "S.Pe.C.", "S.Po.C.", "CINGULATE.", "S.F.int.-F.C.M.ant.",
           "S.F.inf.-BROCA-S.Pe.C.inf.", "S.T.s.", "Sc.Cal.-S.Li.",
           "F.C.M.post.-S.p.C.", "S.T.i.-S.O.T.lat.",
           "OCCIPITAL", "F.I.P.", "S.F.inter.-S.F.sup.",
           "S.F.median-S.F.pol.tr.-S.F.sup.", "S.Or.",
           "S.Or.-S.Olf.", "F.P.O.-S.Cu.-Sc.Cal.",
           "S.s.P.-S.Pa.int.", "S.T.s.br.",
           "Lobule_parietal_sup.", "S.F.marginal-S.F.inf.ant.",
           "F.Coll.-S.Rh.", "S.T.i.-S.T.s.-S.T.pol.",
           "F.C.L.p.-subsc.-F.C.L.a.-INSULA.", "S.F.int.-S.R.",
           "fronto-parietal_medial_face.",
           "S.T.s.-S.GSM.", "F.C.L.p.-S.GSM."
           ]
"""
regions = ["F.I.P."]
sides = ["R"]
input_types = ["skeleton", "foldlabel", "extremities"]
verbose = "-v"


for region in regions:
    for dataset in datasets:
        # loads a already existing template
        pipeline_json = f"{path_dataset_root}/{dataset}/pipeline_loop_2mm.json"
        with open(pipeline_json, 'r') as file:
            json_dict = json.load(file)
            file.close()
        # change the parameters that need to be changed
        # (region, side, input_type)
        json_dict["region_name"] = region
        for side in sides:
            json_dict["side"] = side
            for input_type in input_types:
                json_dict["input_type"] = input_type

                if region == "CINGULATE.":
                    json_dict["combine_type"] = True
                else:
                    json_dict["combine_type"] = False

                if ((region == "OCCIPITAL") or
                        (region == "F.C.L.p.-subsc.-F.C.L.a.-INSULA.")):
                    json_dict["threshold"] = 1
                elif ((region == "F.C.L.p.-subsc.-F.C.L.a.-INSULA.") and
                      (side == "L")):
                    json_dict["threshold"] = 1
                else:
                    json_dict["threshold"] = 0

                # replace the template json by the modified one
                with open(pipeline_json, "w") as file2:
                    json.dump(json_dict, file2, indent=3)
                    file2.close()

                # run the pipeline on the target region
                # with requested parameters read from new json
                if subprocess.call(
                        ["python3", "pipeline.py",
                         "--params_path", f"{pipeline_json}",
                         f"{verbose}"]) != 0:
                    raise ValueError("Error in pipeline: "
                                     "see above for error explanations")
                print("\nEND")
                os.system("which python3")
                print(pipeline_json)
                print(region, dataset, side, input_type, "ok")
                print("\n")
