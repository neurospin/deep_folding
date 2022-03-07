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
import six
from deep_folding.brainvisa import _ALL_SUBJECTS
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import LogJson
from deep_folding.brainvisa.utils.logs import log_command_line
from deep_folding.brainvisa.utils.referentials import \
    ICBM2009c_to_aims_talairach
from deep_folding.brainvisa.utils.referentials import \
    generate_ref_volume_MNI_2009
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.config.logs import set_root_logger_level
from deep_folding.config.logs import set_file_logger
from deep_folding.config.logs import set_file_log_handler
from soma import aims

# Defines logger
log = set_file_logger(__file__)

# Default directory in which lies the manually segmented database
_SRC_DIR_DEFAULT = "/neurospin/dico/data/bv_databases/human/pclean/all"

# Default directory to which we write the masks
_MASK_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/mask"

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'R'

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'F.C.M.ant.'

# Gives the relative path to the manually labelled graph .arg
# in the supervised database
_PATH_TO_GRAPH_DEFAULT = "t1mri/t1/default_analysis/folds/3.3/base2018_manual"


def create_mask(out_voxel_size: tuple) -> aims.Volume:
    """Creates aims volume in MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip

    Args:
        output_voxel_size: tuple
            Output voxel size (default: None, no resampling)

    Returns:
        mask: volume (aims.Volume_S16) filled with 0 in MNI2009 referential
            and with requested voxel_size
    """

    return generate_ref_volume_MNI_2009(out_voxel_size)


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
                    [g_to_icbm_template.transform(np.array(voxel) * voxel_size_in)
                        for voxel in bucket[0].keys()])
                if voxels_real.shape == (0,):
                    continue
                voxels = np.round(np.array(voxels_real) /
                                  voxel_size_out[:3]).astype(int)

                if voxels.shape == (0,):
                    continue
                for i, j, k in voxels:
                    arr[i, j, k, 0] += 1


def write_mask(mask: aims.Volume, mask_file: str):
    """Writes mask on mask file"""
    mask_file_dir = os.path.dirname(mask_file)
    os.makedirs(mask_file_dir, exist_ok=True)
    log.info(f"Final mask file: {mask_file}")
    aims.write(mask, mask_file)


class MaskAroundSulcus:
    """Determines the maximum Bounding Box around given sulci

    It is determined in the  MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip
    """

    def __init__(self,
                 src_dir=_SRC_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 out_voxel_size=None):
        """Inits with list of directories and list of sulci

        Attributes:
            src_dir: list of strings naming full path source directories
            path_to_graph: string naming relative path to labelled graph
            mask_dir: name of target directory with full path
            sulcus: sulcus name
            side: hemisphere side (either L for left, or R for right hemisphere)
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

        self.sulcus = sulcus
        self.mask_dir = mask_dir
        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.voxel_size_out = (out_voxel_size,
                               out_voxel_size,
                               out_voxel_size)

        # Initiliazes mask
        self.mask = aims.Volume()
        self.mask_file = join(
            self.mask_dir,
            self.side,
            self.sulcus + '.nii.gz')

    def get_all_subjects_as_dictionary(self):
        """Lists all subjects from the database (directory src_dir).

        Subjects are the names of the subdirectories of the root directory.

        Returns:
            subjects: a list of dictionaries containing all subjects as dict
        """

        subjects = []

        # Main loop: list all subjects of the directories
        # listed in self.src_dir
        for src_dir, graph_file in zip(self.src_dir, self.graph_file):
            for filename in os.listdir(src_dir):
                directory = os.path.join(src_dir, filename)
                if os.path.isdir(directory):
                    if filename != 'ra':
                        subject = filename
                        subject_d = {
                            'subject': subject,
                            'side': self.side,
                            'dir': src_dir,
                            'graph_file': graph_file % {
                                'side': self.side,
                                'subject': subject}}
                        subjects.append(subject_d)

        return subjects

    def increment_mask(self, subjects: list):
        """Increments mask for the chosen sulcus for all subjects

        Parameters:
            subjects: list containing all subjects to be analyzed
        """

        for sub in subjects:
            log.info(sub)
            # It substitutes 'subject' in graph_file name
            graph_file = sub['graph_file'] % sub
            # It looks for a graph file .arg
            sulci_pattern = glob.glob(join(sub['dir'], graph_file))[0]

            increment_one_mask(sulci_pattern % sub,
                               self.mask,
                               self.sulcus,
                               self.voxel_size_out)

    def compute_mask(self, number_subjects=_ALL_SUBJECTS):
        """Main class program to compute the bounding box

        Args:
            number_subjects: number_subjects to analyze
        """

        if number_subjects:
            subjects = self.get_all_subjects_as_dictionary()

            # Gives the possibility to list only the first number_subjects
            subjects = select_subjects_int(subjects, number_subjects)

            # Creates target mask_dir if they don't exist
            create_folder(self.mask_dir)

            # Creates volume that will take the mask
            self.mask = create_mask(self.voxel_size_out)

            # Increments mask for each sulcus and subjects
            self.increment_mask(subjects)

            # Saving of generated masks
            write_mask(self.mask, self.mask_file)


def mask_around_sulcus(src_dir=_SRC_DIR_DEFAULT,
                       mask_dir=_MASK_DIR_DEFAULT,
                       path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                       sulcus=_SULCUS_DEFAULT,
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
        number_subjects: integer giving the number of subjects to analyze,
            by default it is set to _ALL_SUBJECTS (-1)
        out_voxel_size: float giving voxel size

    Returns:
        aims volume containing the mask
    """

    mask = MaskAroundSulcus(src_dir=src_dir,
                            mask_dir=mask_dir,
                            path_to_graph=path_to_graph,
                            sulcus=sulcus, side=side,
                            out_voxel_size=out_voxel_size)
    mask.compute_mask(number_subjects=number_subjects)

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
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT, nargs='+',
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-m", "--mask_dir", type=str, default=_MASK_DIR_DEFAULT,
        help='Output directory where to store the output mask files. '
             'Default is : ' + _MASK_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box/mask. '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-p", "--path_to_graph", type=str,
        default=_PATH_TO_GRAPH_DEFAULT,
        help='Relative path to manually labelled graph. '
             'Default is ' + _PATH_TO_GRAPH_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose. '
             'Default is : all')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Verbose mode: '
                        'If no option is provided then logging.INFO is selected. '
                        'If one option -v (or -vv) or more is provided '
                        'then logging.DEBUG is selected.')
    parser.add_argument(
        "-x", "--out_voxel_size", type=float, default=None,
        help='Voxel size of of bounding box. '
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)

    # Sets level of root logger
    set_root_logger_level(args.verbose+1)
    # Sets handler for deep_folding logger
    tgt_dir = f"{args.mask_dir}/{args.side}"
    set_file_log_handler(file_dir=tgt_dir,
                         suffix=args.sulcus)

    # Writes command line argument to target dir for logging
    log_command_line(args,
                     prog_name=basename(__file__),
                     tgt_dir=tgt_dir)

    params['src_dir'] = args.src_dir  # src_dir is a list
    params['path_to_graph'] = args.path_to_graph
    # mask_dir is a string, only one directory
    params['mask_dir'] = args.mask_dir
    params['sulcus'] = args.sulcus  # sulcus is a string
    params['side'] = args.side
    params['out_voxel_size'] = args.out_voxel_size

    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


def main(argv):
    """Reads argument line and determines the max bounding box

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        params = parse_args(argv)
        # Actual API
        mask_around_sulcus(src_dir=params['src_dir'],
                           path_to_graph=params['path_to_graph'],
                           mask_dir=params['mask_dir'],
                           sulcus=params['sulcus'],
                           side=params['side'],
                           number_subjects=params['nb_subjects'],
                           out_voxel_size=params['out_voxel_size'])
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
