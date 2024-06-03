"""Uses the pipeline on multiple regions and datasets, for both sides and input type."""

import os
import json
import subprocess


regions = ['S.C.-S.Pe.C.', 'S.Or.']
# datasets = ['candi', 'cnp', 'bsnip1', 'schizconnect-vip-prague']
datasets = ['PreCatatoes']
sides = ['L', 'R']
input_types = ['skeleton', 'foldlabel']


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

                if region == 'CINGULATE.':
                    json_dict['combine_type'] = True
                else:
                    json_dict['combine_type'] = False

                # replace the template json by the modified one
                with open(pipeline_json, 'w') as file2:
                    json.dump(json_dict, file2, indent=3)
                    file2.close()
                
                # run the pipeline on the target region with the requested parameters
                if subprocess.call(["python3", "pipeline.py", "--params_path", f"{pipeline_json}"]) != 0:
                    raise ValueError("Error in pipeline: see above for error explanations")
                print("\nEND")
                os.system("which python3")
                print(pipeline_json)
                print(region, dataset, side, input_type, 'ok')
                print("\n")