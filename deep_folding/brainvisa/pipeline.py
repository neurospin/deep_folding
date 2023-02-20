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

from deep_folding.brainvisa.compute_mask import compute_mask
from deep_folding.brainvisa.generate_crops import generate_crops
from deep_folding.brainvisa.generate_distmaps import generate_distmaps
from deep_folding.brainvisa.generate_foldlabels import generate_foldlabels
from deep_folding.brainvisa.generate_ICBM2009c_transforms import generate_ICBM2009c_transforms
from deep_folding.brainvisa.generate_skeletons import generate_skeletons
from deep_folding.brainvisa.resample_files import resample_files



# Defines logger
log = set_file_logger(__file__)


# get all the sulci of a given brain region
def get_sulci_list(region_name, side, json_path='/neurospin/dico/data/deep_folding/current/sulci_regions_overlap.json'):
    with open(json_path, 'r') as file:
        brain_regions = json.load(file)
    
    if side == 'R':
        side_full = '_right'
    elif side == 'L':
        side_full = '_left'
    else:
        raise ValueError(f"Side argument with an inadequate value. Should be in 'R' or 'L', but is {side}")

    try:
        full_name = region_name + side_full
        sulci_list = list(brain_regions['brain'][full_name].keys())
        for i, sulcus in enumerate(sulci_list):
            sulci_list[i] = sulcus.replace(side_full, '')
    except ValueError:
        print(f"The given region {region_name} is not in the dictonary at {json_path}")

    return sulci_list


def print_info(step, description):
    log.info("\n\n# ----------------------\n"
             f"# STEP {step}: {description}\n"
             "# ----------------------\n")
    return step+1


def is_step_to_be_computed(path, log_string, save_behavior='best'):
    if ((not os.path.exists(path)) or \
        (os.path.exists(path) and os.listdir(path)==[]) or \
        (save_behavior=='best') or \
        (save_behavior=='clear_and_compute')):
        # if no output directory or
        # empty output directory or
        # recompute missing subjects
        # or clear the directory and recompute everything
        return True
    elif save_behavior=='minimal':
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
        raise ValueError(f"Unknown value for save_behavior: {save_behavior}. Choose between 'best' (recommanded)\
minimal and clear_and_compute. Refer to the README for more information.")


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

    args = parser.parse_args(argv)

    set_root_logger_level(args.verbose+1)

    params_path = args.params_path

    with open(params_path, 'r') as file: 
        params = json.load(file)

    return params


@exception_handler
def main(argv):
    """Main function to compute the pipeline, i.e. to compute crops from graphs.

    Args:
        argv: a list containing command line arguments which are
            - params_path: path to the json file where the parameters for the functions called
              by pipeline are stored.
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
        if params['out_voxel_size'] - int(params['out_voxel_size']) == 0: # if voxel size is an int
            vox_size = str(int(params['out_voxel_size']))+'mm'
        else:
            vox_size = str(params['out_voxel_size'])+'mm'
    log.info(vox_size)

    if params['no_mask'] == True:
        mask_str = 'no_mask'
    else:
        mask_str = 'mask'
    
    src_filename = f"{params['input_type']}_" \
        if params['input_type']=="foldlabel" else \
        f"{params['input_type']}_generated_"

    output_filename = f"resampled_{params['input_type']}_"

    if params['input_type'] == 'distmap':
        cropdir_name = "distmap"
    elif params['input_type'] == 'foldlabel':
        cropdir_name = "label"
    else:
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
    params['distmaps_dir'] = os.path.join(params["output_dir"], "distmaps")
    params['foldlabel_dir'] = os.path.join(params["output_dir"], "foldlabels")
    params['transform_dir'] = os.path.join(params["output_dir"], "transforms")
    params['crops_dir'] = os.path.join(params["output_dir"], "crops")


    step = 1
    # generate masks
    step = print_info(step, "generate masks")
    for sulcus in sulci_list:
        log.info(f"Treating the mask generation of {sulcus} (if required).")
        path_to_sulcus_mask = os.path.join(params['masks_dir'], vox_size,
                                           params['side'], sulcus+full_side)
        log.debug(path_to_sulcus_mask)
        
        # never remove the mask folder -> do it by hand if you really want to
        if is_step_to_be_computed(\
                path=path_to_sulcus_mask,
                log_string=f"Mask with the given parameters (side={params['side']}, "
                           f"sulcus={sulcus}, voxel size={vox_size})", 
                save_behavior='best'):
            
            # set up the right parameters
            args_compute_mask = {'src_dir': params['labeled_subjects_dir'],
                                 'path_to_graph': params['path_to_graph_supervised'],
                                 'mask_dir': os.path.join(params['masks_dir'], vox_size),
                                 'sulcus': sulcus,
                                 'new_sulcus': params['new_sulcus'],
                                 'side': params['side'],
                                 'number_subjects': params['nb_subjects_mask'],
                                 'out_voxel_size': params['out_voxel_size'],
                                 'disk_orientation': params['disk_orientation']}
            
            # write the logs as if the command line compute_mask.py was executed
            new_sulcus = args_compute_mask['new_sulcus'] if args_compute_mask['new_sulcus'] else args_compute_mask['sulcus']
            setup_log(Namespace(**{'verbose': log.level, **args_compute_mask}),
                      log_dir=f"{args_compute_mask['mask_dir']}",
                      prog_name='pipeline_compute_masks.py',
                      suffix=complete_sulci_name(new_sulcus, args_compute_mask['side']))
            
            # execute the actual function
            compute_mask(**args_compute_mask)
            log.info('Mask generated')


    # generate raw skeletons
    if params['input_type'] in ['skeleton', 'distmap']:
        step = print_info(step, "generate raw skeletons")
        skel_dir = os.path.join(params['skeleton_dir'], 'raw', params['side'])

        if is_step_to_be_computed(skel_dir, "Raw skeletons", save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(skel_dir):
                # remove the target folder
                log.info(f"Delete {skel_dir}")
                shutil.rmtree(skel_dir)

            args_generate_skeletons = {'src_dir': params['graphs_dir'],
                                    'skeleton_dir': params['skeleton_dir'] + '/raw',
                                    'path_to_graph': params['path_to_graph'],
                                    'side': params['side'],
                                    'junction': params['junction'],
                                    'bids': params['bids'],
                                    'parallel': params['parallel'],
                                    'number_subjects': params['nb_subjects'],
                                    'qc_path': params['skel_qc_path'],
                                    'disk_orientation': params['disk_orientation']}
            
            setup_log(Namespace(**{'verbose': log.level, **args_generate_skeletons}),
                    log_dir=f"{args_generate_skeletons['skeleton_dir']}",
                    prog_name='pipeline_generate_skeletons.py',
                    suffix=full_side[1:])

            generate_skeletons(**args_generate_skeletons)
            log.info('Skeletons generated')
    

    # generate raw distmaps if required
    if params['input_type'] == 'distmap':
        step = print_info(step, "generate raw distmaps")
        distmap_raw_path = os.path.join(params['distmaps_dir'], 'raw', params['side'])

        if is_step_to_be_computed(distmap_raw_path, "Raw distmaps", save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(distmap_raw_path):
                # remove the target folder
                log.info(f"Delete {distmap_raw_path}")
                shutil.rmtree(distmap_raw_path)

            args_generate_distmaps = {'src_dir': params['skeleton_dir'] + '/raw',
                                      'distmaps_dir': params['distmaps_dir'] + '/raw',
                                      'side': params['side'],
                                      'parallel': params['parallel'],
                                      'disk_orientation': params['disk_orientation'],
                                      'resampled_skel': params['resampled_skel'],
                                      'number_subjects': params['nb_subjects']}

            setup_log(Namespace(**{'verbose': log.level, **args_generate_distmaps}),
                      log_dir=f"{args_generate_distmaps['distmaps_dir']}",
                      prog_name='pipeline_generate_distamps.py',
                      suffix=full_side[1:])

            generate_distmaps(**args_generate_distmaps)
            log.info('Raw distmaps generated')

    
    # generate raw foldlabels if required
    if params['input_type'] == 'foldlabel':
        step = print_info(step, "generate raw foldlabels")
        foldlabel_raw_path = os.path.join(params['foldlabel_dir'], 'raw', params['side'])

        if is_step_to_be_computed(foldlabel_raw_path, "Raw foldlabels", save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(foldlabel_raw_path):
                # remove the target folder
                log.info(f"Delete {foldlabel_raw_path}")
                shutil.rmtree(foldlabel_raw_path)
            
            args_generate_foldlabels = {'src_dir':  params['graphs_dir'],
                                        'foldlabel_dir': params['foldlabel_dir'] + '/raw',
                                        'path_to_graph': params['path_to_graph'],
                                        'side': params['side'],
                                        'junction': params['junction'],
                                        'bids': params['bids'],
                                        'parallel': params['parallel'],
                                        'disk_orientation': params['disk_orientation'],
                                        'number_subjects': params['nb_subjects'],
                                        'qc_path': params['skel_qc_path']}

            setup_log(Namespace(**{'verbose': log.level, **args_generate_foldlabels}),
                      log_dir=f"{args_generate_foldlabels['foldlabel_dir']}",
                      prog_name='pipeline_generate_foldlabels.py',
                      suffix=full_side[1:])

            generate_foldlabels(**args_generate_foldlabels)
            log.info('Raw foldlabels generated')


    # generate transform
    if params['out_voxel_size'] != 'raw':
        step = print_info(step, "generate transforms")
        path_to_transforms = os.path.join(params['transform_dir'], params['side'])

        if is_step_to_be_computed(path_to_transforms, "Transforms", save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(path_to_transforms):
                # remove the target folder
                log.info(f"Delete {path_to_transforms}")
                shutil.rmtree(path_to_transforms)
            
            args_generate_transforms = {'src_dir': params['graphs_dir'],
                                        'transform_dir': params['transform_dir'],
                                        'path_to_graph': params['path_to_graph'],
                                        'side': params['side'],
                                        'bids': params['bids'],
                                        'parallel': params['parallel'],
                                        'disk_orientation': params['disk_orientation'],
                                        'number_subjects': params['nb_subjects']}

            setup_log(Namespace(**{'verbose': log.level, **args_generate_transforms}),
                      log_dir=f"{args_generate_transforms['transform_dir']}",
                      prog_name='pipeline_generate_ICBM2009c_transforms.py',
                      suffix=full_side[1:])

            generate_ICBM2009c_transforms(**args_generate_transforms)
            log.info('Transforms generated')


    # resample files
    if params['out_voxel_size'] != 'raw':
        step = print_info(step, f"resample {params['input_type']} files")
        if params['input_type'] == 'distmap':
            raw_input = os.path.join(params['distmaps_dir'], 'raw')
            resampled_dir = os.path.join(params['distmaps_dir'], vox_size)
        
        elif params['input_type'] == 'foldlabel':
            raw_input = os.path.join(params['foldlabel_dir'], 'raw')
            resampled_dir = os.path.join(params['foldlabel_dir'], vox_size)
        
        else:
            # raw data supposed to be skeletons by default
            raw_input = os.path.join(params['skeleton_dir'], 'raw')
            resampled_dir = os.path.join(params['skeleton_dir'], vox_size)
        
        path_resampled_path = os.path.join(resampled_dir, params['side'])

        if is_step_to_be_computed(
                path=path_resampled_path,
                log_string=f"Resampled {params['input_type']}s",
                save_behavior=save_behavior):
            if save_behavior == 'clear_and_compute' and os.path.exists(path_resampled_path):
                # remove the target folder
                log.info(f"Delete {path_resampled_path}")
                shutil.rmtree(path_resampled_path)
            
            args_resample_files = {'src_dir': raw_input,
                                   'input_type': params['input_type'],
                                   'resampled_dir': resampled_dir,
                                   'transform_dir': params['transform_dir'],
                                   'side': params['side'],
                                   'number_subjects': params['nb_subjects'],
                                   'out_voxel_size': params['out_voxel_size'],
                                   'parallel': params['parallel'],
                                   'src_filename': src_filename,
                                   'output_filename': output_filename,
                                   'disk_orienttion': params['disk_orientation']}

            setup_log(Namespace(**{'verbose': log.level, **args_resample_files}),
                      log_dir=f"{args_resample_files['resampled_dir']}",
                      prog_name='pipeline_resample_files.py',
                      suffix=full_side[1:])

            resample_files(**args_resample_files)
            log.info(f"{params['input_type']} resampled")
    

    # generate crops
    if params['input_type'] == 'distmap':
        raw_input = params['distmaps_dir']
        resampled_dir = os.path.join(params['distmaps_dir'], vox_size)
    elif params['input_type'] == 'foldlabel':
        raw_input = params['foldlabel_dir']
        resampled_dir = os.path.join(params['foldlabel_dir'], vox_size)
    else:
        # raw data supposed to be skeletons by default
        raw_input = params['skeleton_dir']
        resampled_dir = os.path.join(params['skeleton_dir'], vox_size)

    
    if params['out_voxel_size'] == 'raw':
        src_dir = raw_input + 'raw'
    else:
        src_dir = resampled_dir
    
    path_to_crops = os.path.join(params['crops_dir'], vox_size, params['region_name'],
                                 mask_str)
    path_to_crops_complete = path_to_crops+'/'+params['side']+cropdir_name+'s'

    step = print_info(step, f"generate {params['input_type']} crops")
    if is_step_to_be_computed(
                path=path_to_crops_complete,
                log_string=f"Crops",
                save_behavior=save_behavior):
        if save_behavior == 'clear_and_compute' and os.path.exists(path_to_crops_complete):
            # remove the target folder
            log.info(f"Delete {path_to_crops_complete}")
            shutil.rmtree(path_to_crops_complete)

        args_generate_crops = {'src_dir': src_dir,
                               'input_type': params['input_type'],
                               'crop_dir': path_to_crops,
                               'bbox_dir': params['bbox_dir'],
                               'mask_dir': params['masks_dir'] + f'/{vox_size}',
                               'side': params['side'],
                               'list_sulci': sulci_list,
                               'cropping_type': params['cropping_type'],
                               'combine_type': params['combine_type'],
                               'parallel': params['parallel'],
                               'number_subjects': params['nb_subjects'],
                               'disk_orientation': params['disk_orientation'],
                               'no_mask': params['no_mask']}
        
        setup_log(Namespace(**{'verbose': log.level, **args_generate_crops}),
                  log_dir=f"{args_generate_crops['crop_dir']}",
                  prog_name='pipeline_generate_crops.py',
                  suffix=full_side[1:]+'_'+args_generate_crops['input_type'])
        
        generate_crops(**args_generate_crops)
        log.info('Crops generated')

        # save params json where the crops lie
        with open(path_to_crops+f'/pipeline_params_{cropdir_name}s.json', 'w') as file:
            json.dump(params, file, indent=2)
    


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
