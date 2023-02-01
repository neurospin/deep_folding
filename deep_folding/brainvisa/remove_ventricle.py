#!python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
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

""" Remove ventricle from a skeleton through the automatic labelled graph
by Morphologist.
Be careful : the script will not work if there are '_' in subject ids
(for database in BIDS format).
"""

import argparse
import glob
import re
import sys
from os.path import abspath
from os.path import basename
from os.path import join
from os.path import exists
from os.path import isdir

from soma import aims
import numpy as np

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected
from pqdm.processes import pqdm
from deep_folding.config.logs import set_file_logger

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, \
    _SKELETON_DIR_DEFAULT, _SIDE_DEFAULT, \
    _PATH_TO_GRAPH_DEFAULT

_OUTPUT_DIR_DEFAULT = join(_SKELETON_DIR_DEFAULT, "without_ventricle")
_SKELETON_FILENAME_DEFAULT = "skeleton_generated"
_OUTPUT_FILENAME_DEFAULT = "skeleton_generated_without_ventricle"

# Defines logger
log = set_file_logger(__file__)


def remove_ventricle_from_one_skeleton(skeleton_file, labelled_graph_file):
    skeleton = aims.read(skeleton_file)
    arr_skel = np.asarray(skeleton)
    labelled_graph = aims.read(labelled_graph_file)
    for vertex in labelled_graph.vertices():
        if "label" in vertex:
            label = vertex["label"]
            if label.startswith("ventricle"):
                for bucket_name, value in {'aims_other': 100,
                                           'aims_ss': 60, 'aims_bottom': 30}.items():
                    bucket = vertex.get(bucket_name)
                    if bucket is not None:
                        voxels = np.array(bucket[0].keys())
                        if voxels.shape == (0,):
                            continue
                        for i, j, k in voxels:
                            arr_skel[i, j, k] = 0
    return skeleton


def parse_args(argv):
    """Parses command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description="Generate skeleton files without ventricle.")
    parser.add_argument(
        "-s", "--skeleton_dir", type=str,
        help="Directory where are the skeletons you want ventricle to be removed from")
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_OUTPUT_DIR_DEFAULT,
        help="Output directory where to put skeleton files without ventricle. "
             f"Default is : {_OUTPUT_DIR_DEFAULT}")
    parser.add_argument(
        "-m", "--morpho_dir", type=str, default=_SRC_DIR_DEFAULT,
        help="Directory where the graph data lies. It has to point directly to"
             "the morphologist directory containing the subjects as subdirectories. "
             f"Default is : {_SRC_DIR_DEFAULT}")
    parser.add_argument(
        "-p", "--path_to_graph", type=str, default=_PATH_TO_GRAPH_DEFAULT,
        help="Relative path to graph. "
             "In BIDS format, the session_acquisition_run directory have to be replaced by *. "
             f"Default is :  {_PATH_TO_GRAPH_DEFAULT}")
    parser.add_argument(
        "-f", "--skeleton_filename", type=str, default=_SKELETON_FILENAME_DEFAULT,
        help="Filename of skeleton files. "
             "Format is : <SIDE><skeleton_filename>_<SUBJECT>.nii.gz "
             f"Default is : {_SKELETON_FILENAME_DEFAULT}")
    parser.add_argument(
        "-e", "--output_filename", type=str, default=_OUTPUT_FILENAME_DEFAULT,
        help="Filename of output files. "
             "Format is : <SIDE><output_filename>_<SUBJECT>.nii.gz "
             f"Default is : {_OUTPUT_FILENAME_DEFAULT}")
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help="Hemisphere side (either L, R or F)"
             f"Default is : {_SIDE_DEFAULT}")
    parser.add_argument(
        "-b", "--bids", default=False, action="store_true",
        help="if the database uses BIDS format"
             "Default is : False")
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')

    args = parser.parse_args(argv)
    dico_suffix = {"R": "right", "L": "left", "F": "full"}
    setup_log(args,
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix=dico_suffix[args.side])

    params = vars(args)

    params['output_dir'] = abspath(args.output_dir)
    params['morpho_dir'] = abspath(args.morpho_dir)
    params['skeleton_dir'] = abspath(args.skeleton_dir)
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    return params


class RemoveVentricleFromSkeleton:
    """Class to remove ventricle from skeleton files through the automatic
    labelling graph computed by Morphologist. The default automatic labelling
    session is : deepcnn_session_auto.

    It contains all the information to get labelled graphs from skeleton
    filenames and to write new sketletons without ventricle in the target
    directory.
    """

    def __init__(self, skeleton_dir, output_dir,
                 morpho_dir, path_to_graph,
                 skeleton_filename, output_filename,
                 side, bids, parallel):

        self.side = side
        self.bids = bids
        self.parallel = parallel
        self.morpho_dir = morpho_dir
        self.path_to_graph = path_to_graph
        self.skeleton_dir = join(skeleton_dir, self.side)
        self.output_dir = join(output_dir, self.side)
        create_folder(abspath(self.output_dir))

        self.expr = f"{self.side}{skeleton_filename}_(.*).nii.gz$"
        self.skeleton_file = join(self.skeleton_dir,
                                  f"%(side)s{skeleton_filename}_%(subject)s.nii.gz")
        self.output_file = join(self.output_dir,
                                f"%(side)s{output_filename}_%(subject)s.nii.gz")

    def remove_ventricle_from_one_subject(self, subject: str):
        """ Removes ventricle and writes new skeleton for one subject.
        """

        sbj = {"subject": subject, "side": self.side}

        skeleton_file = join(self.skeleton_dir, self.skeleton_file % sbj)

        output_file = join(self.output_dir, self.output_file % sbj)

        labelled_graph_list = self.get_labelled_graph(subject)

        log.debug(f"skeleton_file = {skeleton_file}")
        log.debug(f"labelled_graphs = {labelled_graph_list}")
        log.debug(f"output_file = {output_file}")

        if exists(skeleton_file):
            for graph_file in labelled_graph_list:
                if exists(graph_file):
                    skeleton = remove_ventricle_from_one_skeleton(skeleton_file,
                                                                  graph_file)
                else:
                    raise FileNotFoundError(f"Labelled graph not found : \
                                            {graph_file}")
            aims.write(skeleton, output_file)
        else:
            raise FileNotFoundError(f"Skeleton file not found : \
                                    {skeleton_file}")

    def get_labelled_graph(self, subject: str):
        """ Find the labelled graph in the morphologist database from the
        skeleton filename.
        """
        labelled_graph_list = []
        side_list = ["L", "R"] if self.side == "F" else [self.side]
        labelling_session = "deepcnn_session_auto"
        if subject.startswith("_"):
            subject = subject[1:]
        for side in side_list:
            if self.bids:
                split = subject.split("_")
                subject_id = split[0]
                if len(split) > 1:
                    keys = "_".join(split[1:])
                filename = f"{side}{subject_id}_{labelling_session}.arg"
                labelled_graph_file = join(self.morpho_dir, subject_id,
                                           self.path_to_graph.replace("*", keys),
                                           labelling_session, filename)
            else:
                filename = f"{side}{subject}_{labelling_session}.arg"
                labelled_graph_file = join(self.morpho_dir, subject, self.path_to_graph,
                                           labelling_session, filename)
            labelled_graph_list.append(labelled_graph_file)
        return labelled_graph_list

    def compute(self, number_subjects):
        """Loops over subjects and remove ventricle from skeletons.
        """
        # Gets list of subjects
        log.debug(f"skeleton_dir = {self.skeleton_dir}")
        log.debug(f"reg exp = {self.expr}")

        if isdir(self.skeleton_dir):
            filenames = glob.glob(f"{self.skeleton_dir}/*.nii.gz")
            log.debug(f"Skeleton files list = {filenames}")

            list_subjects = [re.search(self.expr, basename(filename))[1]
                             for filename in filenames
                             if not re.search('.minf$', filename)]
            list_subjects = select_subjects_int(list_subjects, number_subjects)

        log.info(f"Expected number of subjects = {len(list_subjects)}")
        log.info(f"list_subjects[:5] = {list_subjects[:5]}")
        log.debug(f"list_subjects = {list_subjects}")

        # Performs computation on all subjects either serially or in parallel
        if self.parallel:
            log.info(
                "PARALLEL MODE: subjects are computed in parallel.")
            pqdm(list_subjects,
                 self.remove_ventricle_from_one_subject,
                 n_jobs=define_njobs())
        else:
            log.info(
                "SERIAL MODE: subjects are scanned serially, "
                "without parallelism")
            for sub in list_subjects:
                self.remove_ventricle_from_one_subject(sub)

        # Checks if there is expected number of generated files
        compare_number_aims_files_with_expected(self.output_dir, list_subjects)


def remove_ventricle(skeleton_dir=_SKELETON_DIR_DEFAULT,
                     output_dir=_OUTPUT_DIR_DEFAULT,
                     morpho_dir=_SRC_DIR_DEFAULT,
                     path_to_graph=_PATH_TO_GRAPH_DEFAULT,
                     skeleton_filename=_SKELETON_FILENAME_DEFAULT,
                     output_filename=_OUTPUT_FILENAME_DEFAULT,
                     side=_SIDE_DEFAULT,
                     bids=False,
                     parallel=False,
                     number_subjects=_ALL_SUBJECTS):
    """Remove ventricle from a skeleton
    through the automatic labelled graph by Morphologist"""

    # Initialization
    removal = RemoveVentricleFromSkeleton(
        skeleton_dir=skeleton_dir,
        output_dir=output_dir,
        morpho_dir=morpho_dir,
        path_to_graph=path_to_graph,
        skeleton_filename=skeleton_filename,
        output_filename=output_filename,
        side=side,
        bids=bids,
        parallel=parallel)
    # Actual generation of skeletons from graphs
    removal.compute(number_subjects=number_subjects)


@exception_handler
def main(argv):
    """Reads argument line and remove ventricle from skeleton
    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    remove_ventricle(
        skeleton_dir=params["skeleton_dir"],
        output_dir=params["output_dir"],
        morpho_dir=params["morpho_dir"],
        path_to_graph=params["path_to_graph"],
        skeleton_filename=params["skeleton_filename"],
        output_filename=params["output_filename"],
        side=params["side"],
        bids=params["bids"],
        parallel=params['parallel'],
        number_subjects=params['nb_subjects'])


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])