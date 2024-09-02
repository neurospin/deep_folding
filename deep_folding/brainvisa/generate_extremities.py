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

"""Write extremities from graph files.

  It writes only the inflated lateral edges of each branch

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python generate_extremities.py


"""

import argparse
import glob
import re
import sys
from os.path import abspath
from os.path import basename
from p_tqdm import p_map

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.subjects import \
    get_number_subjects, is_it_a_subject
from deep_folding.brainvisa.utils.subjects import \
    select_subjects_int, select_good_qc
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.skeleton_extremities import \
    generate_extremities_from_graph_file
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_subjects
from deep_folding.config.logs import set_file_logger

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SRC_DIR_DEFAULT, \
    _EXTREMITIES_DIR_DEFAULT, _SIDE_DEFAULT, \
    _PATH_TO_GRAPH_DEFAULT, \
    _PATH_TO_SKELETON_WITH_HULL_DEFAULT, \
    _QC_PATH_DEFAULT

# Defines logger
log = set_file_logger(__file__)


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
        description='Generates extremity volume files from graphs')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where the graph data lies. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_EXTREMITIES_DIR_DEFAULT,
        help='Output directory where to put skeleton files.'
        'Default is : ' + _EXTREMITIES_DIR_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-p", "--path_to_graph", type=str,
        default=_PATH_TO_GRAPH_DEFAULT,
        help='Relative path to graph. '
             'Default is ' + _PATH_TO_GRAPH_DEFAULT)
    parser.add_argument(
        "-t", "--path_to_skeleton_with_hull", type=str,
        default=_PATH_TO_SKELETON_WITH_HULL_DEFAULT,
        help='Relative path to skeleton with hull. '
             'Default is ' + _PATH_TO_SKELETON_WITH_HULL_DEFAULT)
    parser.add_argument(
        "-q", "--qc_path", type=str,
        default=_QC_PATH_DEFAULT,
        help='Path to quality check .csv. '
             'Default is ' + _QC_PATH_DEFAULT)
    parser.add_argument(
        "-b", "--bids", default=False, action="store_true",
        help="if the database uses the BIDS format"
    )
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

    setup_log(args,
              log_dir=f"{args.output_dir}",
              prog_name=basename(__file__),
              suffix='right' if args.side == 'R' else 'left')

    params = vars(args)

    params['src_dir'] = abspath(args.src_dir)
    params['extremities_dir'] = abspath(args.output_dir)
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    # Removes renamed params
    # So that we can use params dictionary directly as function arguments
    params.pop('output_dir')
    params.pop('verbose')

    return params


class GraphConvert2Extremity:
    """Class to convert all graphs from a folder into extremities

    It contains all information to scan a dataset for graphs
    and for skeleton with hull
    and writes extremities as volume into target directory
    """

    def __init__(self, src_dir, extremities_dir,
                 side, parallel,
                 path_to_skeleton_with_hull,
                 path_to_graph, bids, qc_path):
        self.src_dir = src_dir
        self.extremities_dir = extremities_dir
        self.side = side
        self.qc_path = qc_path
        self.parallel = parallel
        self.path_to_graph = path_to_graph
        self.path_to_skeleton_with_hull = path_to_skeleton_with_hull
        self.bids = bids
        self.extremities_dir = f"{self.extremities_dir}/{self.side}"
        create_folder(abspath(self.extremities_dir))

    def get_extremity_filename(self, subject, graph_file):
        """Creates name of extremity file for one subject"""
        extremity_file = f"{self.extremities_dir}/" + \
            f"{self.side}extremities_{subject}"
        if self.bids:
            session = re.search("ses-([^_/]+)", graph_file)
            acquisition = re.search("acq-([^_/]+)", graph_file)
            run = re.search("run-([^_/]+)", graph_file)
            if session:
                extremity_file += f"_{session[0]}"
            if acquisition:
                extremity_file += f"_{acquisition[0]}"
            if run:
                extremity_file += f"_{run[0]}"
        extremity_file += ".nii.gz"
        return extremity_file

    def generate_one_extremity(self, subject: str):
        """Generates and writes extremity volume for one subject.
        """
        graph_path = f"{self.src_dir}/{subject}/" +\
                     f"{self.path_to_graph}/{self.side}*.arg"
        skeleton_with_hull_path = f"{self.src_dir}/{subject}/" +\
            f"{self.path_to_skeleton_with_hull}/{self.side}skeleton*.nii.gz"

        # Gets graph file path
        list_graph_file = glob.glob(graph_path)
        log.debug(f"list_graph_file = {list_graph_file}")

        # Gets skeleton with hull file path
        list_skeleton_with_hull_file = glob.glob(skeleton_with_hull_path)
        log.debug(
            f"list_skeleton_with_hull_file = {list_skeleton_with_hull_file}")

        if len(list_graph_file) == 0:
            raise RuntimeError(f"No graph file! "
                               f"{graph_path} does not exist")
        if len(list_skeleton_with_hull_file) == 0:
            raise RuntimeError(
                f"No skeleton_with_hull file! "
                f"{skeleton_with_hull_path} does not exist "
                f"or does not contain {self.side}skeleton files")
        if len(list_graph_file) != len(list_skeleton_with_hull_file):
            raise RuntimeError(
                "Different number of graph files "
                "and skeleton with hull files! "
                f"Graph files = {list_graph_file}. "
                f"Skeleton with hull files = {list_skeleton_with_hull_file}")

        for graph_file, skeleton_with_hull_file in \
                zip(list_graph_file, list_skeleton_with_hull_file):
            extremity_file = self.get_extremity_filename(subject, graph_file)
            log.debug(f"skeleton_with_hull_file = {skeleton_with_hull_file}")

            generate_extremities_from_graph_file(graph_file,
                                                 skeleton_with_hull_file,
                                                 extremity_file)
            if not self.bids:
                break

    def compute(self, nb_subjects):
        """Loops over subjects and converts graphs into skeletons.
        """
        # Gets list of subjects
        filenames = glob.glob(f"{self.src_dir}/*")
        list_subjects = [basename(filename) for filename in filenames
                         if is_it_a_subject(filename)]
        log.info(f"Number of subjects before qc = {len(list_subjects)}")
        list_subjects = select_good_qc(list_subjects, self.qc_path)
        not_processed_subjects = \
            get_not_processed_subjects(list_subjects,
                                       self.extremities_dir,
                                       "extremities_")

        list_subjects = select_subjects_int(list_subjects,
                                            not_processed_subjects,
                                            nb_subjects)

        log.info(f"Expected number of subjects = {len(list_subjects)}")
        log.info(f"list_subjects[:5] = {list_subjects[:5]}")
        log.debug(f"list_subjects = {list_subjects}")

        # Performs computation on all subjects either serially or in parallel
        if self.parallel:
            log.info(
                "PARALLEL MODE: subjects are computed in parallel.")
            p_map(self.generate_one_extremity,
                  list_subjects,
                  num_cpus=define_njobs())
        else:
            log.info(
                "SERIAL MODE: subjects are scanned serially, "
                "without parallelism")
            for sub in list_subjects:
                self.generate_one_extremity(sub)

        # Checks if there is expected number of generated files
        if self.bids:
            list_graphs = \
                [g for g in glob.glob(f"{self.src_dir}/*/{self.path_to_graph}")
                 if not re.search('.minf$', g)]
            compare_number_aims_files_with_expected(self.extremities_dir,
                                                    list_graphs)
        else:
            compare_number_aims_files_with_expected(self.extremities_dir,
                                                    list_subjects)


def generate_extremities(
        src_dir=_SRC_DIR_DEFAULT,
        extremities_dir=_EXTREMITIES_DIR_DEFAULT,
        path_to_skeleton_with_hull=_PATH_TO_SKELETON_WITH_HULL_DEFAULT,
        path_to_graph=_PATH_TO_GRAPH_DEFAULT,
        side=_SIDE_DEFAULT,
        bids=False,
        parallel=False,
        nb_subjects=_ALL_SUBJECTS,
        qc_path=_QC_PATH_DEFAULT):
    """Generates extremities (inflated lmateral edges) from graphs"""

    # Gets function arguments and values
    params = locals()
    nb_subjects = params.pop('nb_subjects')

    # Initialization with same arguments and values as function
    conversion = GraphConvert2Extremity(**params)
    # Actual generation of skeletons from graphs
    conversion.compute(nb_subjects=nb_subjects)


@exception_handler
def main(argv):
    """Reads argument line and generates skeleton from graph

    Args:
        argv: a list containing command line arguments
    """
    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    generate_extremities(**params)


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
