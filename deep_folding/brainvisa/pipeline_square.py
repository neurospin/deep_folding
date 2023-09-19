"""Uses the pipeline on multiple regions and datasets, for both sides and input type."""

import os
import json


regions = ['S.F.inter.', 'S.F.inf.-BROCA-S.Pe.C.inf.', 'S.T.i.', 'S.C.', 'F.C.M.post.-S.p.C.']
datasets = ['bsnip1', 'candi', 'cnp', 'schizconnect-vip-prague']
sides = ['R', 'L']
input_types = ['foldlabel', 'skeleton']


for region in regions:
    for dataset in datasets:
        # loads a already existing template
        pipeline_json = f"/neurospin/dico/data/deep_folding/current/datasets/{dataset}/pipeline_loop_2mm.json"
        with open(pipeline_json, 'r') as file:
            json_dict = json.load(file)
            file.close()
        # change the parameters that need to be changed (region, side, input_type)
        json_dict['region_name'] = region
        for side in sides:
            json_dict['side'] = side
            for input_type in input_types:
                json_dict['input_type'] = input_type

                # replace the template json by the modified one
                with open(pipeline_json, 'w') as file2:
                    json.dump(json_dict, file2)
                    file2.close()
                
                # run the pipeline on the target region with the requested parameters
                os.system(f"python3 brainvisa/pipeline.py --params_path {pipeline_json}")
                print("\nEND")
                print(region, dataset, side, input_type, 'ok')
                print("\n")