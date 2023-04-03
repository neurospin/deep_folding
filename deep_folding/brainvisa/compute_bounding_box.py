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
The aim of this script is to compute the bounding box for a given sulcus
based on a manually labelled dataset

Bounding box corresponds to the biggest box that encompasses the given sulci
on all subjects of the manually labelled dataset. It measures the bounding box
in the MNI152  space.
"""
import argparse
import glob
import sys
from os.path import basename
from os.path import join

import numpy as np
from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.bbox import compute_max
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import LogJson
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.referentials import \
    ICBM2009c_to_aims_talairach
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.subjects import \
    get_all_subjects_as_dictionary
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.config.logs import set_file_logger
from soma import aims

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SUPERVISED_SRC_DIR_DEFAULT,\
    _BBOX_DIR_DEFAULT, _SIDE_DEFAULT, _SULCUS_DEFAULT,\
    _VOXEL_SIZE_DEFAULT, _PATH_TO_GRAPH_SUPERVISED_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def box_ICBM2009c_to_aims_talairach(bbmin_mni152: np.array,
                                    bbmax_mni152: np.array) -> tuple:
    """Transform bbox coordinates from MNI152 to AIMS talairach referential"""

    bbmin_tal = ICBM2009c_to_aims_talairach(bbmin_mni152)
    bbmax_tal = ICBM2009c_to_aims_talairach(bbmax_mni152)
    log.debug(f"box (AIMS Talairach) min: {bbmin_tal}")
    log.debug(f"box (AIMS Talairach) max: {bbmax_tal}")

    return bbmin_tal, bbmax_tal


def get_one_bounding_box(graph_filename, sulcus):
    """get bounding box of the chosen sulcus for one data graph in MNI 152

    Function that outputs the bounding box for the listed sulci
    for this datagraph. The bounding box is the smallest rectangular box
    that encompasses the chosen sulcus.
    It is given in the MNI 152 referential.

    Args:
        graph_filename: string being the name of graph file .arg to analyze:
                        for example: 'Lammon_base2018_manual.arg'

    Returns:
        bbox_min: numpy array giving the upper right vertex coordinates
                of the box in the MNI 152 referential
        bbox_max: numpy array fiving the lower left vertex coordinates
                of the box in the MNI 152 referential
    """

    # Reads the data graph and transforms it to AIMS Talairach referential
    # Note that this is NOT the MNI Talairach referential
    # This is the Talairach referential used in AIMS
    # There are several Talairach referentials
    graph = aims.read(graph_filename)
    voxel_size_in = graph['voxel_size'][:3]
    g_to_icbm_template = \
        aims.GraphManip.getICBM2009cTemplateTransform(graph)
    bbox_min = None
    bbox_max = None

    # Gets the min and max coordinates of the sulci
    # by looping over all the vertices of the graph
    for vertex in graph.vertices():
        vname = vertex.get('name')
        if vname != sulcus:
            continue
        for bucket_name in ('aims_ss', 'aims_bottom', 'aims_other'):
            bucket = vertex.get(bucket_name)
            if bucket is not None:
                voxels = np.asarray(
                    [g_to_icbm_template.transform(np.array(voxel)
                                                  * voxel_size_in)
                        for voxel in bucket[0].keys()])
                if voxels.shape == (0,):
                    continue

                bbox_min = np.min(np.vstack(
                    ([bbox_min] if bbox_min is not None else [])
                    + [voxels]), axis=0)
                bbox_max = np.max(np.vstack(
                    ([bbox_max] if bbox_max is not None else [])
                    + [voxels]), axis=0)

    log.debug(f"box (MNI 152) min: {bbox_min}")
    log.debug(f"box (MNI 152) max: {bbox_max}")

    return bbox_min, bbox_max


def get_bounding_boxes(subjects, sulcus):
    """get bounding boxes of the chosen sulcus for all subjects.

    Function that outputs bounding box for the listed sulci on a manually
    labeled dataset.
    Bounding box corresponds to the biggest box encountered in the manually
    labeled subjects in the MNI1 152 space.
    The bounding box is the smallest rectangular box that
    encompasses the sulcus.

    Args:
        subjects: list containing all subjects to be analyzed
        sulcus: str -> sulcus to be analyzed

    Returns:
        tuple (list_bbmin, list_bbmax) with:
            list_bbmin: list containing the upper right vertex of the box
                    in the MNI 152 space;
            list_bbmax: list containing the lower left vertex of the box
                    in the MNI 152 space
    """

    # Initialization
    list_bbmin = []
    list_bbmax = []

    for sub in subjects:
        log.info(sub)
        graph_file = sub['graph_file'] % sub
        # It looks for a graph file .arg
        list_graph_file = glob.glob(join(sub['dir'], graph_file))
        if len(list_graph_file) == 0:
            raise RuntimeError(f"No graph file! "
                               f"{sub['dir']} doesn't contain {graph_file}")
        sulci_pattern = list_graph_file[0]

        bbox_min, bbox_max = \
            get_one_bounding_box(sulci_pattern % sub, sulcus)
        if bbox_min is not None:
            list_bbmin.append([bbox_min[0], bbox_min[1], bbox_min[2]])
            list_bbmax.append([bbox_max[0], bbox_max[1], bbox_max[2]])
        else:
            log.debug(
                f"No sulcus {sulcus} found for {sub}; it can be OK.")

    if not list_bbmin:
        raise ValueError(f"No sulcus named {sulcus} found "
                         'for the whole dataset. '
                         'It is an error. You should check sulcus name.')

    return list_bbmin, list_bbmax


def compute_box_voxel(bbmin_mni152, bbmax_mni152, voxel_size_out):
    """Returns the coordinates of the box as voxels

    Coordinates of the box in voxels are determined in the MNI referential

    Args:
        bbmin_mni152: numpy array with coordinates of the upper right corner
                of the box (MNI152 space)
        bbmax_mni152: numpy array with coordinates of the lower left corner
                of the box (MNI152 space)
        voxel_size: voxel size (in MNI referential or HCP normalized SPM space)

    Returns:
        tuple (bbmin_vox, bbmax_vox) with
            bbmin_vox: numpy array with coordinates of the upper right corner
                of the box (voxels in MNI space);
            bbmax_vox: numpy array with coordinates of the lower left corner
                    of the box (voxels in MNI space)
    """

    # To go back from mms to voxels
    voxel_size = voxel_size_out
    bbmin_vox = np.round(np.array(bbmin_mni152) /
                         voxel_size[:3]).astype(int)
    bbmax_vox = np.round(np.array(bbmax_mni152) /
                         voxel_size[:3]).astype(int)

    return bbmin_vox, bbmax_vox


class BoundingBoxMax:
    """Determines the maximum Bounding Box around given sulci

    It is determined in the  MNI ICBM152 nonlinear 2009c asymmetrical template
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09b_nifti.zip
    """

    def __init__(self,
                 src_dir=_SUPERVISED_SRC_DIR_DEFAULT,
                 path_to_graph=_PATH_TO_GRAPH_SUPERVISED_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 sulcus=_SULCUS_DEFAULT,
                 new_sulcus=None,
                 side=_SIDE_DEFAULT,
                 out_voxel_size=None):
        """Inits with list of directories and list of sulci

        Attributes:
            src_dir: list of strings naming full path source directories
            path_to_graph: string naming relative path to labelled graph
            bbox_dir: name of target directory with full path
            sulcus: sulcus name
            new_sulcus: new sulcus name
            side: hemisphere side (L for left, or R for right hemisphere)
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
        self.bbox_dir = bbox_dir
        self.side = side
        self.sulcus = complete_sulci_name(sulcus, side)
        self.new_sulcus = complete_sulci_name(self.new_sulcus, side)
        self.voxel_size_out = (out_voxel_size,
                               out_voxel_size,
                               out_voxel_size)

        # Json full name is the name of the sulcus + .json
        # and is kept under the subdirectory Left or Right
        json_file = join(self.bbox_dir, self.side, self.new_sulcus + '.json')
        self.json = LogJson(json_file)

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

            # Logs general information on json file
            self.json.write_general_info()

            # Gives the possibility to list only the first number_subjects
            subjects = select_subjects_int(subjects, number_subjects)

            # Creates target bbox dir if it doesn't exist
            create_folder(self.bbox_dir)

            # Logs number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(subjects),
                           'src_dir': self.src_dir,
                           'bbox_dir': self.bbox_dir,
                           'out_voxel_size': self.voxel_size_out[0]}
            self.json.update(dict_to_add)

            # MAIN PROGRAM
            # Determines box for each subject
            list_bbmin, list_bbmax = get_bounding_boxes(subjects, self.sulcus)

            # Determines the box encompassing the sulcus for all subjects
            # The coordinates are determined in MNI 152  space
            # And takes the max box in two referentials:
            # ICBM2009c and aims Talairach
            bbmin_mni152, bbmax_mni152 = compute_max(list_bbmin, list_bbmax)

            bbmin_tal, bbmax_tal = \
                box_ICBM2009c_to_aims_talairach(bbmin_mni152, bbmax_mni152)

            # Determines the box encompassing the sulcus for all subjects
            # The coordinates are determined in voxels in ICBM009c space
            bbmin_vox, bbmax_vox = compute_box_voxel(bbmin_mni152,
                                                     bbmax_mni152,
                                                     self.voxel_size_out)

            # Logging results on json file
            dict_to_add = {'side': self.side,
                           'sulcus': self.sulcus,
                           'new_sulcus': self.new_sulcus,
                           'bbmin_voxel': bbmin_vox.tolist(),
                           'bbmax_voxel': bbmax_vox.tolist(),
                           'bbmin_MNI152': bbmin_mni152.tolist(),
                           'bbmax_MNI152': bbmax_mni152.tolist(),
                           'bbmin_AIMS_Talairach': bbmin_tal.tolist(),
                           'bbmax_AIMS_Talairach': bbmax_tal.tolist()
                           }
            self.json.update(dict_to_add)
            log.debug(f"box (voxel): min = {bbmin_vox}")
            log.debug(f"box (voxel): max = {bbmax_vox}")

        else:
            bbmin_vox = 0
            bbmax_vox = 0

        return bbmin_vox, bbmax_vox


def compute_bounding_box(src_dir=_SUPERVISED_SRC_DIR_DEFAULT,
                         bbox_dir=_BBOX_DIR_DEFAULT,
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
        bbox_dir: string giving target bbox directory path
        path_to_graph: string giving relative path to manually labelled graph
        side: hemisphere side (either 'L' for left, or 'R' for right)
        sulcus: string giving the sulcus to analyze
        new_sulcus: string giving a new sulcus name (optional)
        number_subjects: integer giving the number of subjects to analyze,
            by default it is set to _ALL_SUBJECTS (-1)
        out_voxel_size: float giving voxel size in mm

    Returns:
        tuple (bbmin_vox, bbmax_vox)
    """

    box = BoundingBoxMax(src_dir=src_dir,
                         bbox_dir=bbox_dir,
                         path_to_graph=path_to_graph,
                         sulcus=sulcus,
                         new_sulcus=new_sulcus,
                         side=side,
                         out_voxel_size=out_voxel_size)
    bbmin_vox, bbmax_vox = box.compute(
        number_subjects=number_subjects)

    return bbmin_vox, bbmax_vox


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
        description='Computes bounding box around the named sulcus')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SUPERVISED_SRC_DIR_DEFAULT,
        nargs='+',
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SUPERVISED_SRC_DIR_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_BBOX_DIR_DEFAULT,
        help='Output directory where to store the output bbox json files. '
             'Default is : ' + _BBOX_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT,
        help='Sulcus name around which we determine the bounding box. '
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
        help='Voxel size of bounding box. '
             'Default is : None')

    params = {}

    args = parser.parse_args(argv)

    # Sets logger level, files log handler and prints/logs command line
    new_sulcus = args.new_sulcus if args.new_sulcus else args.sulcus
    setup_log(args,
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix=complete_sulci_name(new_sulcus, args.side))

    params['src_dir'] = args.src_dir  # src_dir is a list
    params['path_to_graph'] = args.path_to_graph
    # bbox_dir is a string, only one directory
    params['bbox_dir'] = args.output_dir
    params['sulcus'] = args.sulcus  # sulcus is a string
    params['new_sulcus'] = args.new_sulcus
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
    compute_bounding_box(src_dir=params['src_dir'],
                         path_to_graph=params['path_to_graph'],
                         bbox_dir=params['bbox_dir'],
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
