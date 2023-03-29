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
The aim of this script is to output bounding box and mask for a given sulcus
based on a manually labelled dataset

Bounding box corresponds to the biggest box that encompasses the given sulci
on all subjects of the manually labelled dataset. It measures the bounding box
in the MNI152  space.
"""
import argparse
import glob
import os
import sys
from os.path import basename
from os.path import join

import numpy as np
from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.referentials import \
    generate_ref_volume_ICBM2009c
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.subjects import \
    get_all_subjects_as_dictionary
from deep_folding.brainvisa.utils.quality_checks import \
    get_not_processed_subjects_dict
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.config.logs import set_file_logger
from soma import aims

# Imports constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SUPERVISED_SRC_DIR_DEFAULT,\
    _MASK_DIR_DEFAULT, _SIDE_DEFAULT, _SULCUS_DEFAULT,\
    _VOXEL_SIZE_DEFAULT, _PATH_TO_GRAPH_SUPERVISED_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def initialize_mask(out_voxel_size: tuple) -> aims.Volume:
    """Creates aims volume in MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip

    Args:
        output_voxel_size: tuple
            Output voxel size (default: None, no resampling)

    Returns:
        mask: volume (aims.Volume_S16) filled with 0 in MNI2009 referential
            and with requested voxel_size
    """

    return generate_ref_volume_ICBM2009c(out_voxel_size)


def increment_one_mask(graph_filename, mask, sulcus, voxel_size_out):
    """Increments self.mask of 1 where there is the sulcus

    Args:
        graph_filename: string being the name of graph file .arg to analyze:
                        for example: 'Lammon_base2018_manual.arg'

    """

    # Reads the data graph and transforms it to MNI ICBM152 referential
    graph = aims.read(graph_filename)
    g_to_icbm_template = aims.GraphManip.getICBM2009cTemplateTransform(
        graph)
    voxel_size_in = graph['voxel_size'][:3]
    arr = np.asarray(mask)
    arr_one = np.zeros(arr.shape).astype(np.int16)

    # Gets the min and max coordinates of the sulci
    # by looping over all the vertices of the graph
    for vertex in graph.vertices():
        vname = vertex.get('name')
        if vname != sulcus:
            continue
        for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels_real = np.asarray(
                    [g_to_icbm_template.transform(
                        np.array(voxel) * voxel_size_in)
                        for voxel in bucket[0].keys()])
                if voxels_real.shape == (0,):
                    continue
                voxels = np.round(np.array(voxels_real) /
                                  voxel_size_out[:3]).astype(int)

                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr_one[i, j, k, 0] += 1

    # Guarantees that array is one 1 even if vosel is present twice
    arr_one = (arr_one >= 1).astype(np.int16)

    # Puts in aims volume the array of one mask
    vol_one = aims.Volume(arr_one)
    vol_one.copyHeaderFrom(mask.header())
    vol_one.header()['voxel_size'] = mask.header()['voxel_size']

    # Increments global mask
    arr += arr_one

    return vol_one


def increment_mask(subjects: list,
                   mask: aims.Volume,
                   sulcus: str,
                   voxel_size_out: tuple,
                   sample_dir: str):
    """Increments mask for the chosen sulcus for all subjects

    Parameters:
        subjects: list containing all subjects to be analyzed
    """

    for sub in subjects:
        log.info(sub)
        # It substitutes 'subject' in graph_file name
        graph_file = sub['graph_file'] % sub
        # It looks for a graph file .arg
        list_graph_file = glob.glob(join(sub['dir'], graph_file))
        if len(list_graph_file) == 0:
            raise RuntimeError(f"No graph file! "
                               f"{sub['dir']} doesn't contain {graph_file}")
        sulci_pattern = list_graph_file[0]

        vol_one = increment_one_mask(sulci_pattern % sub,
                                     mask,
                                     sulcus,
                                     voxel_size_out)
        aims.write(vol_one, f"{sample_dir}/{sub['subject']}.nii.gz")


def write_mask(mask: aims.Volume, mask_file: str):
    """Writes mask on mask file"""
    mask_file_dir = os.path.dirname(mask_file)
    os.makedirs(mask_file_dir, exist_ok=True)
    log.info(f"\nFinal mask file: {mask_file}\n")
    aims.write(mask, mask_file)


class MaskAroundSulcus:
    """Determines the maximum Bounding Box around given sulci

    It is determined in the  MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip
    """

    def __init__(self,
                 src_dir=_SUPERVISED_SRC_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_SUPERVISED_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 new_sulcus=None,
                 side=_SIDE_DEFAULT,
                 out_voxel_size=None):
        """Inits with list of directories and list of sulci

        Attributes:
            src_dir: list of strings naming full path source directories
            path_to_graph: string naming relative path to labelled graph
            mask_dir: name of target directory with full path
            sulcus: sulcus name
            new_sulcus: new sulcus name
            side: hemisphere side (either L for left,
                                or R for right hemisphere)
            out_voxel_size: float for voxel size in mm
        """

        # Transforms input source dir to a list of strings
        self.src_dir = [src_dir] if isinstance(src_dir, str) else src_dir

        # manually labelled graph file relative to the subject directory
        # we use the '*' glob to take into account different naming conventions
        # It must be put in the same order as src_dir
        path_to_graph = ([path_to_graph] if isinstance(path_to_graph, str)
                         else path_to_graph)
        self.graph_file = []
        for path in path_to_graph:
            self.graph_file.append('%(subject)s/'
                                   + path
                                   + '/%(side)s%(subject)s*.arg')

        self.new_sulcus = new_sulcus if new_sulcus else sulcus
        self.mask_dir = mask_dir
        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.new_sulcus = complete_sulci_name(self.new_sulcus, side)
        self.mask_sample_dir = f"{mask_dir}/{self.side}/{self.new_sulcus}"
        self.out_voxel_size = (out_voxel_size,
                               out_voxel_size,
                               out_voxel_size)

        # Initiliazes mask
        self.mask = aims.Volume()
        self.mask_file = join(
            self.mask_dir,
            self.side,
            self.new_sulcus + '.nii.gz')

    def compute(self, number_subjects=_ALL_SUBJECTS):
        """Main class program to compute the bounding box

        Args:
            number_subjects: number_subjects to analyze
        """

        if number_subjects:
            subjects = get_all_subjects_as_dictionary(
                self.src_dir,
                self.graph_file,
                self.side)

            # Creates target mask_dir if they don't exist
            create_folder(self.mask_dir)
            create_folder(self.mask_sample_dir)

            # Generates list of subjects not treated yet
            not_treated_subjects = get_not_processed_subjects_dict(
                subjects, self.mask_sample_dir)

            if len(not_treated_subjects):
                # Gives the possibility to list only the first number_subjects
                subjects = select_subjects_int(subjects, number_subjects)

                # Creates volume that will take the mask
                self.mask = initialize_mask(self.out_voxel_size)

                # Increments mask for each sulcus and subjects
                increment_mask(subjects,
                               self.mask,
                               self.sulcus,
                               self.out_voxel_size,
                               self.mask_sample_dir)

                # Saving of generated masks
                write_mask(self.mask, self.mask_file)


def compute_mask(src_dir=_SUPERVISED_SRC_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_SUPERVISED_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 new_sulcus=None,
                 side=_SIDE_DEFAULT,
                 number_subjects=_ALL_SUBJECTS,
                 out_voxel_size=None):
    """ Main program computing the box encompassing the sulcus in all subjects

    The programm loops over all subjects
    and computes in MNI space the voxel coordinates of the box encompassing
    the sulci for all subjects

    Args:
        src_dir: list of strings -> directories of the supervised databases
        mask_dir: string giving target mask directory path
        path_to_graph: string giving relative path to manually labelled graph
        side: hemisphere side (either 'L' for left, or 'R' for right)
        sulcus: string giving the sulcus to analyze
        new_sulcus: string giving a new sulcus name (optional)
        number_subjects: integer giving the number of subjects to analyze,
            by default it is set to _ALL_SUBJECTS (-1)
        out_voxel_size: float giving voxel size

    Returns:
        aims volume containing the mask
    """
    mask = MaskAroundSulcus(src_dir=src_dir,
                            mask_dir=mask_dir,
                            path_to_graph=path_to_graph,
                            sulcus=sulcus,
                            new_sulcus=new_sulcus,
                            side=side,
                            out_voxel_size=out_voxel_size)
    mask.compute(number_subjects=number_subjects)

    return mask.mask


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
        description='Computes mask around the named sulcus')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SUPERVISED_SRC_DIR_DEFAULT,
        nargs='+',
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SUPERVISED_SRC_DIR_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_MASK_DIR_DEFAULT,
        help='Output directory where to store the output mask files. '
             'Default is : ' + _MASK_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box/mask. '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-w", "--new_sulcus", type=str, default=None,
        help='Sulcus name around which we determine the bounding box. '
             'Default is : None (same name as \'sulcus\')')
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-p", "--path_to_graph", type=str,
        default=_PATH_TO_GRAPH_SUPERVISED_DEFAULT,
        help='Relative path to manually labelled graph. '
             'Default is ' + _PATH_TO_GRAPH_SUPERVISED_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose. '
             'Default is : all')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')
    parser.add_argument(
        "-x", "--out_voxel_size", type=float, default=_VOXEL_SIZE_DEFAULT,
        help='Voxel size of mask. '
             'Default is : ' + str(_VOXEL_SIZE_DEFAULT))

    params = {}

    args = parser.parse_args(argv)

    # Sets logger level, fils log handler and prints/logs command line
    new_sulcus = args.new_sulcus if args.new_sulcus else args.sulcus
    setup_log(args,
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix=complete_sulci_name(new_sulcus, args.side))

    params['src_dir'] = args.src_dir  # src_dir is a list
    params['path_to_graph'] = args.path_to_graph
    # mask_dir is a string, only one directory
    params['mask_dir'] = args.output_dir
    params['sulcus'] = args.sulcus  # sulcus is a string
    params['new_sulcus'] = args.new_sulcus  # sulcus is a string
    params['side'] = args.side
    params['out_voxel_size'] = args.out_voxel_size

    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


@exception_handler
def main(argv):
    """Reads argument line and determines the max bounding box

    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)
    # Actual API
    compute_mask(
        src_dir=params['src_dir'],
        path_to_graph=params['path_to_graph'],
        mask_dir=params['mask_dir'],
        sulcus=params['sulcus'],
        new_sulcus=params['new_sulcus'],
        side=params['side'],
        number_subjects=params['nb_subjects'],
        out_voxel_size=params['out_voxel_size'])


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
