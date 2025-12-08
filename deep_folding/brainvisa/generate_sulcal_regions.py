"""
Uses the pipeline on multiple regions and datasets,
for both sides and input type."""

import argparse
import os
from os.path import join
import json
import subprocess
import sys

from deep_folding.brainvisa.utils.folder import get_nth_parent_dir

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.config.logs import set_file_logger

# Defines logger
log = set_file_logger(__file__)

# The relative path leads to right outside of deep_folding directory and /data/ which is the prefered file architecture for accessing the data
#_PATH_DATASET_ROOT_DEFAULT = os.path.join(get_nth_parent_dir(os.getcwd(), 3), 'data/') #"/neurospin/dico/data/deep_folding/current/datasets"
# _DATASETS_DEFAULT = ["UkBioBank40"]
_DEEPFOLDING_VERSION = "2025"
_SIDES_DEFAULT = ["L", "R"]
_INPUT_TYPES_DEFAULT = ["skeleton", "foldlabel", "extremities"]
_REGIONS_DEFAULT = ["S.C.-sylv.", "S.C.-S.Pe.C.", "S.C.-S.Po.C.",\
            "S.Pe.C.", "S.Po.C.", "S.F.int.-F.C.M.ant.",\
            "S.F.inf.-BROCA-S.Pe.C.inf.", "S.T.s.", "Sc.Cal.-S.Li.",\
            "F.C.M.post.-S.p.C.", "S.T.i.-S.O.T.lat.",\
            "OCCIPITAL", "F.I.P.-F.I.P.Po.C.inf.", "S.F.inter.-S.F.sup.",\
            "S.F.median-S.F.pol.tr.-S.F.sup.", "S.Or.",\
            "S.Or.-S.Olf.", "F.P.O.-S.Cu.-Sc.Cal.",\
            "S.s.P.-S.Pa.int.", "S.T.s.br.",\
            "Lobule_parietal_sup.", "S.F.marginal-S.F.inf.ant.",\
            "F.Coll.-S.Rh.", "S.T.i.-S.T.s.-S.T.pol.",\
            "F.C.L.p.-subsc.-F.C.L.a.-INSULA.", "S.F.int.-S.R.",\
            "S.Call.", "S.Call.-S.s.P.-S.intraCing."\
            ]


def parse_args(argv):
    """Function parsing command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__),
        description='Generates all specified sulcal regions')
    parser.add_argument(
        "-d", "--path_dataset", type=str,
        help='Path where deep_folding dataset lie.',
        required=True)
    parser.add_argument(
        "-o", "--output_dir", type=str,
        help="Path where deep_folding derivatives will lie." \
        "Default is $DATASET_PATH/derivatives/."
    )
    parser.add_argument(
        "--path_to_graph", type=str, required=True
    )
    parser.add_argument(
        "--path_sk_with_hull", type=str, required=True
    )
    parser.add_argument(
        "--sk_qc_path", type=str, default=""
    )
    # parser.add_argument(
    #     "-d", "--datasets", type=str, default=_DATASETS_DEFAULT, nargs='+',
    #     help='Datasets to process. '
    #          'Give all desired datasets one after the other. '
    #          'Example: -d dataset1 dataset2'
    #          'Default is : ' + ' '.join(_DATASETS_DEFAULT))
    parser.add_argument(
        "-i", "--sides", type=str, default=_SIDES_DEFAULT, nargs='+',
        help='Hemisphere side (either L or R). '
             'Gives the desired sides one after the other. '
             'Example: -i L R'
             'Default is : ' + ' '.join(_SIDES_DEFAULT))
    parser.add_argument(
        "-y", "--input_types", type=str, default=_INPUT_TYPES_DEFAULT, nargs='+',
        help='Input types: \'skeleton\', \'foldlabel\', \'extremities\'. '
        'Give the desired types one after the other. '
        'Example: -y skeleton foldlabel'
        'Default is : ' + ' '.join(_INPUT_TYPES_DEFAULT))
    parser.add_argument(
        "-r", "--regions", type=str, default=_REGIONS_DEFAULT, nargs='+',
        help='Give desired sulcal regions. '
             'Gives the desired sulcal regions one after the other. '
             'Example: -r S.C.-sylv. S.F.inf.-BROCA-S.Pe.C.inf.'
             'Default is : ' + ' '.join(_REGIONS_DEFAULT))
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')
    parser.add_argument(
        "--njobs", help="Number of CPU cores allowed to use. Default is your maximum number of cores - 2 or up to 22 if you have enough cores.",
        type=int
    )

    params = {}

    args = parser.parse_args(argv)

    params = vars(args)

    verbose = '-' + ('v' * args.verbose) if args.verbose > 0 else ''
    

    # params['crop_dir'] = args.output_dir
    # params['list_sulci'] = args.sulcus  # a list of sulci

    # # Checks if nb_subjects is either the string "all" or a positive integer
    # params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    # # Removes renamed params
    # # So that we can use params dictionary directly as function arguments
    # params.pop('output_dir')
    # params.pop('sulcus')
    params.pop('verbose')
    params['verbose'] = verbose

    return params


def generate_sulcal_regions(regions, sides, input_types,
                            path_dataset, verbose, output_dir, path_to_graph,
                            path_sk_with_hull, sk_qc_path, njobs):
    """Global loops to generate all regions for all dataset"""
    
    
    
    for region in regions:
        
        #for dataset in datasets:
        
        # loads a already existing template
        pipeline_json = f"{path_dataset}/pipeline_loop_2mm.json"
        with open(pipeline_json, 'r') as file:
            json_dict = json.load(file)

            # Modifying templated values in the JSON file
            for k, v in json_dict.items():
                if v == "$local":
                    if k == "brain_regions_json":
                        json_dict[k] = join(get_nth_parent_dir(os.getcwd(), 3), 
                                                                'champollion_pipeline/sulci_regions_champollion_V1.json')
                    if k == "supervised_output_dir":
                        json_dict[k] = join(get_nth_parent_dir(os.getcwd(), 3), 'deep_folding/data')
                    if k == "graphs_dir":
                        json_dict[k] = join(path_dataset, "derivatives/morphologist-5.2")
                    if k == "output_dir":
                        json_dict[k] = join(path_dataset, f"derivatives/deep_folding-{_DEEPFOLDING_VERSION}" 
                                                       if output_dir != "" or output_dir is not None 
                                                       else output_dir)
                    if k == "path_to_graph" and (path_to_graph != "" or path_to_graph is not None):
                        json_dict[k] = path_to_graph
                    if k == "path_to_skeleton_with_hull" and (path_sk_with_hull != "" or path_sk_with_hull is not None):
                        json_dict[k] = path_sk_with_hull
                    if k == "skel_qc_path" and (sk_qc_path != "" or sk_qc_path is not None):
                        json_dict[k] = sk_qc_path
            
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
                if verbose == '':
                    if subprocess.call(
                            ["python3", "generate_one_sulcal_region.py",
                            "--params_path", f"{pipeline_json}", "--njobs", str(njobs)]) != 0:
                        raise ValueError("Error in pipeline: "
                                        "see above for error explanations")
                else:
                    if subprocess.call(
                            ["python3", "generate_one_sulcal_region.py",
                            "--params_path", f"{pipeline_json}",
                            f"{verbose}", "--njobs", str(njobs)]) != 0:
                        raise ValueError("Error in pipeline: "
                                        "see above for error explanations")
                print("\nEND")
                os.system("which python3")
                print(pipeline_json)
                print(region, path_dataset, side, input_type, "ok")
                print("\n")


@exception_handler
def main(argv):
    """Reads argument line and generates sulcal regions
    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)
    print(params)

    # Actual API
    generate_sulcal_regions(**params)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
