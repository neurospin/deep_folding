#!python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
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
"""
Pipeline to build crops from morphologist graph
"""
import json
import os
import argparse
import sys
import shutil

from os.path import basename
from argparse import Namespace


from deep_folding.brainvisa import exception_handler
from deep_folding.config.logs import set_file_logger, set_root_logger_level
from deep_folding.brainvisa.utils.logs import setup_log

from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.brainvisa.utils.folder import get_nth_parent_dir

from deep_folding.brainvisa.compute_mask import compute_mask
from deep_folding.brainvisa.generate_crops import generate_crops
from deep_folding.brainvisa.generate_distbottom_crops import \
    generate_distbottom_crops
from deep_folding.brainvisa.generate_distmaps import generate_distmaps
from deep_folding.brainvisa.generate_foldlabels import generate_foldlabels
from deep_folding.brainvisa.mask_resampled_foldlabels import \
    mask_foldlabel_files
from deep_folding.brainvisa.mask_resampled_extremities import \
    mask_extremities_files
from deep_folding.brainvisa.generate_ICBM2009c_transforms import \
    generate_ICBM2009c_transforms
from deep_folding.brainvisa.generate_skeletons import generate_skeletons
from deep_folding.brainvisa.generate_extremities import generate_extremities
from deep_folding.brainvisa.resample_files import resample_files


# Defines logger
log = set_file_logger(__file__)


# get all the sulci of a given brain region
def get_sulci_list(
        region_name,
        side,
        json_path=os.path.join(
            get_nth_parent_dir(os.path.abspath(__file__), 5),
            'sulci_regions_champollion_V1.json'
        )):
    """Gets list of sulci corresponding to a region"""

    with open(json_path, 'r') as file:
        brain_regions = json.load(file)

    if side == 'R':
        side_full = '_right'
    elif side == 'L':
        side_full = '_left'
    else:
        raise ValueError(
            f"Side argument with an inadequate value. "
            f"Should be in 'R' or 'L', but is {side}")

    try:
        full_name = region_name + side_full
        sulci_list = list(brain_regions['brain'][full_name].keys())
        for i, sulcus in enumerate(sulci_list):
            sulci_list[i] = sulcus.replace(side_full, '')
    except ValueError:
        print(
            f"The given region {region_name} "
            f"is not in the dictionary at {json_path}")

    return sulci_list


def print_info(step, description):
    log.info("\n\n# ----------------------\n"
             f"# STEP {step}: {description}\n"
             "# ----------------------\n")
    return step + 1


def is_directory_empty(path):
    if os.path.isdir(path):
        if os.listdir(path) == []:
            return True
        else:
            return False
    else:
        return False


def is_step_to_be_computed(path, log_string, save_behavior='best'):
    if ((not os.path.exists(path)) or
        (os.path.exists(path) and is_directory_empty(path)) or
        (save_behavior == 'best') or
            (save_behavior == 'clear_and_compute')):
        # if no output directory or
        # empty output directory or
        # recompute missing subjects
        # or clear the directory and recompute everything
        return True
    elif save_behavior == 'minimal':
        log.info(f"{log_string} are already computed. "
                 "If you want to overwrite them, "
                 f"please either delete the folder {path} "
                 "or set 'save_behavior' to 'clear_and_compute'"
                 "if you want to generate it all again. "
                 "Otherwise, if you want to generate "
                 "only not processed subjects, "
                 "set 'save_behavior' to 'best' in json configuration file")
        return False
    else:
        raise ValueError(
            f"Unknown value for save_behavior: {save_behavior}."
            f"Choose between 'best' (recommanded)"
            "minimal and clear_and_compute. "
            "Refer to the README for more information.")


def check_if_same_dim(arr, df):
    assert (arr.shape[0] == len(
        df)), "Number of subjects differs between numpy array and csv"


def parse_args(argv: list) -> dict:
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: a dictionary with all arugments as keys
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description='Apply the entire pipeline from graph to crops.')
    parser.add_argument(
        "--params_path", type=str, default='./params_pipeline.json',
        help='json file where the parameters for the functions called'
             'by pipeline are stored.'
             'Default is: ./params_pipeline.json')
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

    args = parser.parse_args(argv)

    set_root_logger_level(args.verbose + 1)

    params_path = args.params_path

    with open(params_path, 'r') as file:
        params = json.load(file)

    params["njobs"] = args.njobs

    return params


@exception_handler
def main(argv):
    """Main function to compute the pipeline, i.e. to compute crops from graphs

    Args:
        argv: a list containing command line arguments which are
            - params_path: path to the json file where the parameters
                for the functions called by pipeline are stored.
            - verbose: If no option is provided then logging.INFO is selected.
              If one option -v (or -vv) or more is provided
              then logging.DEBUG is selected.
    """

    # Parsing arguments
    params = parse_args(argv)
    log.debug(params)

    # define shortcuts to create folders or run command lines
    if params['side'] == 'R':
        full_side = '_right'
    elif params['side'] == 'L':
        full_side = '_left'

    if params['out_voxel_size'] == 'raw':
        vox_size = 'raw'
    else:
        if params['out_voxel_size'] - \
                int(params['out_voxel_size']) == 0:  # if voxel size is an int
            vox_size = str(int(params['out_voxel_size'])) + 'mm'
        else:
            vox_size = str(params['out_voxel_size']) + 'mm'
    log.info(vox_size)

    if params['no_mask']:
        mask_str = 'no_mask'
    else:
        mask_str = 'mask'

    match params['input_type']:
        case "foldlabel":
            src_filename = f"{params['input_type']}_"
        case "extremities":
            src_filename = f"{params['input_type']}_"
        case _:
            src_filename = f"{params['input_type']}_generated_"

    output_filename = f"resampled_{params['input_type']}_"

    match params['input_type']:
        case 'distmap':
            cropdir_name = "distmap"
        case 'foldlabel':
            cropdir_name = "label"
        case 'extremities':
            cropdir_name = "extremities"
        case _:
            cropdir_name = "crop"

    save_behavior = params['save_behavior']

    # get the concerned sulci
    sulci_list = get_sulci_list(params['region_name'], side=params['side'],
                                json_path=params['brain_regions_json'])
    log.info(sulci_list)

    # Generates supervised output paths
    params['masks_dir'] = os.path.join(params["supervised_output_dir"], "mask")
    params['bbox_dir'] = os.path.join(params["supervised_output_dir"], "bbox")

    # Generates unsupervised output paths
    params['skeleton_dir'] = os.path.join(params["output_dir"], "skeletons")
    params['extremities_dir'] = \
        os.path.join(params["output_dir"], "extremities")
    params['distmaps_dir'] = os.path.join(params["output_dir"], "distmaps")
    params['foldlabel_dir'] = os.path.join(params["output_dir"], "foldlabels")
    params['transform_dir'] = os.path.join(params["output_dir"], "transforms")
    params['crops_dir'] = os.path.join(params["output_dir"], "crops")

    ##########################################################
    # generate masks
    ##########################################################
    step = 1
    step = print_info(step, "generate masks")
    for sulcus in sulci_list:
        log.info(f"Treating the mask generation of {sulcus} (if required).")
        path_to_sulcus_mask = os.path.join(params['masks_dir'], vox_size,
                                           params['side'], sulcus + full_side)
        log.debug(path_to_sulcus_mask)

        # never remove the mask folder -> do it by hand if you really want to
        if is_step_to_be_computed(
                path=path_to_sulcus_mask + ".nii.gz",
                log_string=f"Mask with the given parameters "
                           f"(side={params['side']}, "
                           f"sulcus={sulcus}, voxel size={vox_size})",
                save_behavior='minimal'):

            # set up the right parameters
            args_compute_mask = {
                'src_dir': params['labeled_subjects_dir'],
                'path_to_graph': params['path_to_graph_supervised'],
                'mask_dir': os.path.join(
                    params['masks_dir'],
                    vox_size),
                'sulcus': sulcus,
                'new_sulcus': params['new_sulcus'],
                'side': params['side'],
                'number_subjects': params['nb_subjects_mask'],
                'out_voxel_size': params['out_voxel_size']}

            # write the logs as if the command line compute_mask.py was
            # executed
            new_sulcus = (
                args_compute_mask['new_sulcus']
                if args_compute_mask['new_sulcus']
                else args_compute_mask['sulcus'])

            setup_log(
                Namespace(**{'verbose': log.level,
                          **args_compute_mask}),
                log_dir=f"{args_compute_mask['mask_dir']}",
                prog_name='pipeline_compute_masks.py',
                suffix=complete_sulci_name(new_sulcus,
                                           args_compute_mask['side']))

            # execute the actual function
            compute_mask(**args_compute_mask)
            log.info('Mask generated')

    ##########################################################
    # generate raw volumes
    #  (either skeletons, extremities, foldlabels, or distmaps)
    ##########################################################

    # generate raw skeletons
    if params['input_type'] in ['skeleton', 'distmap']:
        step = print_info(step, "generate raw skeletons")
        skel_dir = os.path.join(params['skeleton_dir'], 'raw', params['side'])

        if is_step_to_be_computed(
            skel_dir,
            "Raw skeletons",
                save_behavior=save_behavior):

            if (save_behavior == 'clear_and_compute' and
                    os.path.exists(skel_dir)):
                # remove the target folder
                log.info(f"Delete {skel_dir}")
                shutil.rmtree(skel_dir)

            args_generate_skeletons = {
                'src_dir': params['graphs_dir'],
                'skeleton_dir': params['skeleton_dir'] + '/raw',
                'path_to_graph': params['path_to_graph'],
                'side': params['side'],
                'junction': params['junction'],
                'bids': params['bids'],
                'parallel': params['parallel'],
                'nb_subjects': params['nb_subjects'],
                'qc_path': params['skel_qc_path'],
                'njobs': params['njobs']}

            setup_log(
                Namespace(**{'verbose': log.level,
                          **args_generate_skeletons}),
                log_dir=f"{args_generate_skeletons['skeleton_dir']}",
                prog_name='pipeline_generate_skeletons.py',
                suffix=full_side[1:])

            generate_skeletons(**args_generate_skeletons)
            log.info('Skeletons generated')

    # generate raw distmaps if required
    if params['input_type'] == 'distmap':
        step = print_info(step, "generate raw distmaps")
        distmap_raw_path = os.path.join(
            params['distmaps_dir'], 'raw', params['side'])

        if is_step_to_be_computed(
            distmap_raw_path,
            "Raw distmaps",
                save_behavior=save_behavior):

            if (save_behavior == 'clear_and_compute' and
                    os.path.exists(distmap_raw_path)):
                # remove the target folder
                log.info(f"Delete {distmap_raw_path}")
                shutil.rmtree(distmap_raw_path)

            args_generate_distmaps = {
                'src_dir': params['skeleton_dir'] + '/raw',
                'distmaps_dir': params['distmaps_dir'] + '/raw',
                'side': params['side'],
                'parallel': params['parallel'],
                'resampled_skel': params['resampled_skel'],
                'nb_subjects': params['nb_subjects']}

            setup_log(
                Namespace(**{'verbose': log.level,
                          **args_generate_distmaps}),
                log_dir=f"{args_generate_distmaps['distmaps_dir']}",
                prog_name='pipeline_generate_distamps.py',
                suffix=full_side[1:])

            generate_distmaps(**args_generate_distmaps)
            log.info('Raw distmaps generated')

    # generate raw extremities if required
    if params['input_type'] == 'extremities':
        step = print_info(step, "generate raw extremities")
        extremities_raw_path = os.path.join(
            params['foldlabel_dir'], 'raw', params['side'])

        if is_step_to_be_computed(
            extremities_raw_path,
            "Raw extremities",
                save_behavior=save_behavior):

            if (save_behavior == 'clear_and_compute' and
                    os.path.exists(extremities_raw_path)):
                #  remove the target folder
                log.info(f"Delete {extremities_raw_path}")
                shutil.rmtree(extremities_raw_path)

            args_generate_extremities = {
                'src_dir': params['graphs_dir'],
                'extremities_dir': params['extremities_dir'] + '/raw',
                'path_to_skeleton_with_hull':
                    params['path_to_skeleton_with_hull'],
                'path_to_graph': params['path_to_graph'],
                'side': params['side'],
                'bids': params['bids'],
                'parallel': params['parallel'],
                'nb_subjects': params['nb_subjects'],
                'qc_path': params['skel_qc_path']}

            setup_log(
                Namespace(**{'verbose': log.level,
                          **args_generate_extremities}),
                log_dir=f"{args_generate_extremities['extremities_dir']}",
                prog_name='pipeline_generate_extremities.py',
                suffix=full_side[1:])

            generate_extremities(**args_generate_extremities)
            log.info('Raw extremities generated')

    # generate raw foldlabels if required
    if params['input_type'] == 'foldlabel':
        step = print_info(step, "generate raw foldlabels")
        foldlabel_raw_path = os.path.join(
            params['foldlabel_dir'], 'raw', params['side'])

        if is_step_to_be_computed(
            foldlabel_raw_path,
            "Raw foldlabels",
                save_behavior=save_behavior):
            if (save_behavior == 'clear_and_compute' and
                    os.path.exists(foldlabel_raw_path)):
                #  remove the target folder
                log.info(f"Delete {foldlabel_raw_path}")
                shutil.rmtree(foldlabel_raw_path)

            args_generate_foldlabels = {
                'src_dir': params['graphs_dir'],
                'foldlabel_dir': params['foldlabel_dir'] + '/raw',
                'path_to_graph': params['path_to_graph'],
                'side': params['side'],
                'junction': params['junction'],
                'bids': params['bids'],
                'parallel': params['parallel'],
                'nb_subjects': params['nb_subjects'],
                'qc_path': params['skel_qc_path']}

            setup_log(Namespace(**{'verbose': log.level,
                                   **args_generate_foldlabels}),
                      log_dir=f"{args_generate_foldlabels['foldlabel_dir']}",
                      prog_name='pipeline_generate_foldlabels.py',
                      suffix=full_side[1:])

            generate_foldlabels(**args_generate_foldlabels)
            log.info('Raw foldlabels generated')

    ##########################################################
    # generate transform
    ##########################################################
    if params['out_voxel_size'] != 'raw':
        step = print_info(step, "generate transforms")
        path_to_transforms = os.path.join(
            params['transform_dir'], params['side'])

        if is_step_to_be_computed(
            path_to_transforms,
            "Transforms",
                save_behavior=save_behavior):

            if (save_behavior == 'clear_and_compute' and
                    os.path.exists(path_to_transforms)):
                # remove the target folder
                log.info(f"Delete {path_to_transforms}")
                shutil.rmtree(path_to_transforms)

            args_generate_transforms = {
                'src_dir': params['graphs_dir'],
                'transform_dir': params['transform_dir'],
                'path_to_graph': params['path_to_graph'],
                'side': params['side'],
                'bids': params['bids'],
                'parallel': params['parallel'],
                'nb_subjects': params['nb_subjects'],
                'qc_path': params['skel_qc_path']}

            setup_log(
                Namespace(**{'verbose': log.level,
                             **args_generate_transforms}),
                log_dir=f"{args_generate_transforms['transform_dir']}",
                prog_name='pipeline_generate_ICBM2009c_transforms.py',
                suffix=full_side[1:])

            generate_ICBM2009c_transforms(**args_generate_transforms)
            log.info('Transforms generated')

    ##########################################################
    # resample files
    ##########################################################
    if params['out_voxel_size'] != 'raw':
        step = print_info(step, f"resample {params['input_type']} files")
        if params['input_type'] == 'distmap':
            raw_input = os.path.join(params['distmaps_dir'], 'raw')
            resampled_dir = os.path.join(params['distmaps_dir'], vox_size)

        elif params['input_type'] == 'foldlabel':
            raw_input = os.path.join(params['foldlabel_dir'], 'raw')
            resampled_dir = os.path.join(params['foldlabel_dir'], vox_size)

        elif params['input_type'] == 'extremities':
            raw_input = os.path.join(params['extremities_dir'], 'raw')
            resampled_dir = os.path.join(params['extremities_dir'], vox_size)

        else:
            # raw data supposed to be skeletons by default
            raw_input = os.path.join(params['skeleton_dir'], 'raw')
            resampled_dir = os.path.join(params['skeleton_dir'], vox_size)

        # if foldlabel, we generate foldlabels before masking with skeletons
        path_resampled_path = os.path.join(resampled_dir, params['side'])
        if params['input_type'] == 'foldlabel':
            path_resampled_path = path_resampled_path + "_before_masking"
        elif params['input_type'] == 'extremities':
            path_resampled_path = path_resampled_path + "_before_masking"

        if is_step_to_be_computed(
                path=path_resampled_path,
                log_string=f"Resampled {params['input_type']}s",
                save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(
                    path_resampled_path):
                # remove the target folder
                log.info(f"Delete {path_resampled_path}")
                shutil.rmtree(path_resampled_path)

            args_resample_files = {'src_dir': raw_input,
                                   'input_type': params['input_type'],
                                   'resampled_dir': resampled_dir,
                                   'transform_dir': params['transform_dir'],
                                   'side': params['side'],
                                   'nb_subjects': params['nb_subjects'],
                                   'out_voxel_size': params['out_voxel_size'],
                                   'parallel': params['parallel'],
                                   'src_filename': src_filename,
                                   'output_filename': output_filename}

            setup_log(
                Namespace(**{'verbose': log.level,
                             **args_resample_files}),
                log_dir=f"{args_resample_files['resampled_dir']}",
                prog_name='pipeline_resample_files.py',
                suffix=full_side[1:])

            resample_files(**args_resample_files)
            log.info(f"{params['input_type']} resampled")

        # Mask foldlabels with reskeletized skeletons
        if params['input_type'] == "foldlabel":
            masked_dir = os.path.join(params['foldlabel_dir'], vox_size)
            skeleton_dir = os.path.join(params['skeleton_dir'], vox_size)
            path_masked_path = os.path.join(masked_dir, params['side'])

            # check_if_number_skeletons_equals_number_foldlabels(
            #     resampled_dir, skeleton_dir
            # )

            if is_step_to_be_computed(
                    path=path_masked_path,
                    log_string=f"Masking {params['input_type']}s",
                    save_behavior=save_behavior):
                if save_behavior == 'clear_and_compute' and os.path.exists(
                        path_masked_path):
                    # remove the target folder
                    log.info(f"Delete {path_masked_path}")
                    shutil.rmtree(path_masked_path)

                args_masked_files = {'src_dir': masked_dir,
                                     'skeleton_dir': skeleton_dir,
                                     'masked_dir': masked_dir,
                                     'side': params['side'],
                                     'nb_subjects': params['nb_subjects'],
                                     'parallel': params['parallel']}

                setup_log(Namespace(**{'verbose': log.level,
                                    **args_masked_files}),
                          log_dir=f"{args_masked_files['masked_dir']}",
                          prog_name='pipeline_mask_resampled_foldlabels.py',
                          suffix=full_side[1:])

                mask_foldlabel_files(**args_masked_files)
                log.info(f"{params['input_type']} masked")

        # Mask extremities with reskeletized skeletons
        if params['input_type'] == "extremities":
            masked_dir = os.path.join(params['extremities_dir'], vox_size)
            skeleton_dir = os.path.join(params['skeleton_dir'], vox_size)
            path_masked_path = os.path.join(masked_dir, params['side'])

            if is_step_to_be_computed(
                    path=path_masked_path,
                    log_string=f"Masking {params['input_type']}s",
                    save_behavior=save_behavior):
                if save_behavior == 'clear_and_compute' and os.path.exists(
                        path_masked_path):
                    # remove the target folder
                    log.info(f"Delete {path_masked_path}")
                    shutil.rmtree(path_masked_path)

                args_masked_files = {'src_dir': masked_dir,
                                     'skeleton_dir': skeleton_dir,
                                     'masked_dir': masked_dir,
                                     'side': params['side'],
                                     'nb_subjects': params['nb_subjects'],
                                     'parallel': params['parallel']}

                setup_log(Namespace(**{'verbose': log.level,
                                    **args_masked_files}),
                          log_dir=f"{args_masked_files['masked_dir']}",
                          prog_name='pipeline_mask_resampled_extremities.py',
                          suffix=full_side[1:])

                mask_extremities_files(**args_masked_files)
                log.info(f"{params['input_type']} masked")

    ##########################################################
    # generate crops
    ##########################################################
    if params['input_type'] == 'distmap':
        raw_input = params['distmaps_dir']
        resampled_dir = os.path.join(params['distmaps_dir'], vox_size)
    elif params['input_type'] == 'foldlabel':
        raw_input = params['foldlabel_dir']
        resampled_dir = os.path.join(params['foldlabel_dir'], vox_size)
    elif params['input_type'] == 'extremities':
        raw_input = params['extremities_dir']
        resampled_dir = os.path.join(params['extremities_dir'], vox_size)
    else:
        # raw data supposed to be skeletons by default
        raw_input = params['skeleton_dir']
        resampled_dir = os.path.join(params['skeleton_dir'], vox_size)

    if params['out_voxel_size'] == 'raw':
        src_dir = raw_input + 'raw'
    else:
        src_dir = resampled_dir

    path_to_crops = os.path.join(
        params['crops_dir'],
        vox_size,
        params['region_name'],
        mask_str)
    path_to_crops_complete = path_to_crops + \
        '/' + params['side'] + cropdir_name + 's'

    step = print_info(step, f"generate {params['input_type']} crops")
    if is_step_to_be_computed(
            path=path_to_crops_complete,
            log_string="Crops",
            save_behavior=save_behavior):
        if save_behavior == 'clear_and_compute' and os.path.exists(
                path_to_crops_complete):
            # remove the target folder
            log.info(f"Delete {path_to_crops_complete}")
            shutil.rmtree(path_to_crops_complete)

        args_generate_crops = {
            'src_dir': src_dir,
            'input_type': params['input_type'],
            'crop_dir': path_to_crops,
            'bbox_dir': params['bbox_dir'],
            'mask_dir': params['masks_dir'] + f'/{vox_size}',
            'side': params['side'],
            'list_sulci': sulci_list,
            'cropping_type': params['cropping_type'],
            'combine_type': params['combine_type'],
            'parallel': params['parallel'],
            'nb_subjects': params['nb_subjects'],
            'no_mask': params['no_mask'],
            'threshold': params['threshold'],
            'dilation': params['dilation'],
            'njobs': params['njobs']}

        setup_log(Namespace(**{'verbose': log.level, **args_generate_crops}),
                  log_dir=f"{args_generate_crops['crop_dir']}",
                  prog_name='pipeline_generate_crops.py',
                  suffix=full_side[1:] + '_' +
                  args_generate_crops['input_type'])

        generate_crops(**args_generate_crops)
        log.info('Crops generated')

        # save params json where the crops lie
        with open(path_to_crops +
                  f"/pipeline_params_{params['side']}{cropdir_name}s.json",
                  'w') as file:
            json.dump(params, file, indent=2)

    ##########################################################
    # generate distbottom crops
    ##########################################################
    if params['input_type'] == 'skeleton':
        path_to_distbottom_complete = path_to_crops + \
            '/' + params['side'] + "distbottom"

        step = print_info(step, "generate distbottom crops")
        if is_step_to_be_computed(
                path=path_to_distbottom_complete,
                log_string="Distbottoms",
                save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(
                    path_to_crops_complete):
                # remove the target folder
                log.info(f"Delete {path_to_distbottom_complete}")
                shutil.rmtree(path_to_distbottom_complete)

            args_generate_distbottom = {
                'src_dir': path_to_crops,
                'crop_dir': path_to_crops,
                'side': params['side'],
                'parallel': params['parallel'],
                'nb_subjects': params['nb_subjects']}

            setup_log(
                Namespace(**{'verbose': log.level,
                             **args_generate_distbottom}),
                log_dir=f"{args_generate_distbottom['crop_dir']}",
                prog_name='generate_distbottom_crops.py',
                suffix=full_side[1:])

            generate_distbottom_crops(**args_generate_distbottom)
            log.info('Crops generated')

            # save params json where the crops lie
            with open(path_to_crops +
                      f"/pipeline_params_{params['side']}{cropdir_name}s.json",
                      'w') as file:
                json.dump(params, file, indent=2)


######################################################################
# Main program
######################################################################
if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
