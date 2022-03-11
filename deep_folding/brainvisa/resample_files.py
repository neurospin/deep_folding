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

"""Resample skeletons

The aim of this script is to resample skeletons.

  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python resample_files.py

  Alternatively, you can launch the script in the interactive terminal ipython:
  >>> %run resample_files.py

"""

import argparse
import glob
import os
import re
import sys
import tempfile
from os.path import abspath
from os.path import join
from os.path import basename

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.resample import resample
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.brainvisa.utils.logs import setup_log
from pqdm.processes import pqdm
from deep_folding.config.logs import set_file_logger
from soma import aims

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _SKELETON_DIR_DEFAULT,\
    _TRANSFORM_DIR_DEFAULT, _SIDE_DEFAULT,\
    _SULCUS_DEFAULT, _VOXEL_SIZE_DEFAULT

# Defines logger
log = set_file_logger(__file__)

# temporary directory
temp_dir = tempfile.mkdtemp()


def resample_one_file(input_image,
                      out_voxel_size,
                      transformation):
    """Resamples one skeleton

    Args
    ----
        input_image: either string or aims.Volume
            either path to skeleton or skeleton aims Volume
        out_voxel_size: tuple
            Output voxel size (default: None, no resampling)
        transformation: string or aims.Volume
            either path to transformation file or transformation itself

    Returns:
        resampled: aims.Volume
            Transformed or resampled volume
    """

    # We give values with ascendent priority
    # The more important is the inversion in the priority
    # for the bottom value (30) and the simple surface value (60)
    # with respect to the natural order
    # We don't give background, which is the interior 0
    values = np.array([11, 60, 30, 10, 20, 40, 50, 70, 80, 90])

    # Normalization and resampling of skeleton images
    resampled = resample(input_image=input_image,
                         output_vs=out_voxel_size,
                         transformation=transformation,
                         values=values)
    return resampled


class FileResampler:
    """Resamples all files from source directories
    """

    def __init__(self,
                 graph_dir=_GRAPH_DIR_DEFAULT,
                 src_dir=_SRC_DIR_DEFAULT,
                 tgt_dir=_TGT_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 out_voxel_size=_VOXEL_SIZE_DEFAULT):
        """Inits with list of directories and list of sulci

        Args:
            graph_dir: list of strings naming full path source directories,
                    containing MRI and graph images
            src_dir: folder containing generated skeletons and labels
            tgt_dir: name of target (output) directory with full path
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left, or R for right hemisphere)
        """

        self.graph_dir = graph_dir
        self.src_dir = src_dir
        self.side = side
        # Transforms sulcus in a list of sulci
        self.list_sulci = ([list_sulci] if isinstance(list_sulci, str)
                           else list_sulci)
        self.list_sulci = complete_sulci_name(self.list_sulci, self.side)
        self.tgt_dir = tgt_dir
        self.morphologist_dir = morphologist_dir
        self.out_voxel_size = out_voxel_size

        # Morphologist directory
        self.morphologist_dir = join(self.graph_dir, self.morphologist_dir)

        # default acquisition subdirectory
        self.acquisition_dir = "%(subject)s/t1mri/default_acquisition"

        # Names of files in function of dictionary: keys -> 'subject' and 'side'
        # Generated skeleton from folding graphs
        self.skeleton_dir = join(self.src_dir, 'skeleton', self.side)
        self.skeleton_file = join(
            self.skeleton_dir,
            '%(side)sskeleton_generated_%(subject)s.nii.gz')
        self.foldlabel_dir = join(self.src_dir, 'foldlabel', self.side)
        self.foldlabel_file = join(self.foldlabel_dir,
                                   '%(side)sfoldlabel_%(subject)s.nii.gz')

        self.graph_file = 'default_analysis/folds/3.1/default_session_auto/' \
            '%(side)s%(subject)s_default_session_auto.arg'

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.resampled_skeleton_dir = join(self.tgt_dir, 'skeleton', self.side)
        self.resampled_skeleton_file = join(
            self.resampled_skeleton_dir,
            '%(subject)s_resampled_skeleton.nii.gz')
        self.resampled_label_dir = join(self.tgt_dir, 'foldlabel', self.side)
        self.resampled_label_file = join(self.resampled_label_dir,
                                         '%(subject)s_resampled_label.nii.gz')

        # Initialization of bounding box coordinates
        self.bbmin = np.zeros(3)
        self.bbmax = np.zeros(3)

        # Creates json log class
        json_file = join(self.tgt_dir, self.side + 'dataset.json')
        self.json = LogJson(json_file)

        # reference file in MNI template with correct voxel size
        self.ref_file = f"{temp_dir}/file_ref.nii.gz"

    def resample_one_file(self, subject_id, verbose=False):
        """Resamples one skeleton file and one foldlabel file

        Args:
            subject_id: string giving the subject ID
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        subject = {'subject': subject_id, 'side': self.side}

        # Names directory where input subject analysis files are stored
        subject_dir = \
            join(self.morphologist_dir, self.acquisition_dir % subject)

        # Creates transformation MNI template
        file_graph = join(subject_dir, self.graph_file % subject)
        graph = aims.read(file_graph)
        g_to_icbm_template = aims.GraphManip.getICBM2009cTemplateTransform(
            graph)

        # Input skeleton file name
        file_skeleton = self.skeleton_file % {
            'subject': subject_id, 'side': self.side}
        # Input foldlabel file name
        file_foldlabel = self.foldlabel_file % {
            'subject': subject_id, 'side': self.side}

        if os.path.exists(file_skeleton):
            # Creates output (cropped) file name
            file_resampled_skeleton = self.resampled_skeleton_file % {
                'subject': subject_id, 'side': self.side}

            resampled = resample_one_file(
                input_image=file_skeleton,
                out_voxel_size=self.out_voxel_size,
                transformation=g_to_icbm_template)
            aims.write(resampled, file_resampled_skeleton)

        else:
            raise FileNotFoundError(f"{file_skeleton} not found")

    def compute(self, number_subjects=_ALL_SUBJECTS):
        """Loops over nii files

        The programm loops over all subjects from the input (source) directory.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if number_subjects:

            # subjects are detected as the nifti file names under src_dir
            expr = '^.skeleton_generated_([0-9a-zA-Z]*).nii.gz$'
            if os.path.isdir(self.skeleton_dir):
                list_all_subjects = [re.search(expr, os.path.basename(dI))[1]
                                     for dI in glob.glob(f"{self.skeleton_dir}/*.nii.gz")]
            else:
                raise NotADirectoryError(
                    f"{self.sksleton_dir} doesn't exist or is not a directory")

            # Gives the possibility to list only the first number_subjects
            list_subjects = (
                list_all_subjects
                if number_subjects == _ALL_SUBJECTS
                else list_all_subjects[:number_subjects])

            # Creates target directories
            if not os.path.exists(self.resampled_skeleton_dir):
                os.makedirs(self.resampled_skeleton_dir)
            if not os.path.exists(self.resampled_label_dir):
                os.makedirs(self.resampled_label_dir)

            # Writes number of subjects and directory names to json file
            dict_to_add = {'nb_subjects': len(list_subjects),
                           'graph_dir': self.graph_dir,
                           'src_dir': self.src_dir,
                           'side': self.side,
                           'tgt_dir': self.tgt_dir,
                           'skeleton_dir': self.skeleton_dir,
                           'label_dir': self.foldlabel_dir,
                           'resampling_type': 'sulcus-based',
                           'out_voxel_size': self.out_voxel_size
                           }
            self.json.update(dict_to_add=dict_to_add)

            # Performs resampling for each file in a parallelized way
            log.info("list_subjects[:5] = ", list_subjects[:5])
            log.debug("list_subjects = ", list_subjects)

            if self.parallel:
                log.info(
                    "PARALLEL MODE: subjects are in parallel")
                pqdm(
                    list_subjects,
                    self.resample_one_file,
                    n_jobs=define_njobs())
           else:
                log.info(
                    "SERIAL MODE: subjects are scanned serially")
                 for sub in list_subjects:
                    self.resample_one_file(sub)

    def resample_skeletons(self, number_subjects=_ALL_SUBJECTS):
        """Main API to resample skeletons

        The programm loops over all subjects from the input (source) directory.

        Args:
            number_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        self.json.write_general_info()

        # Generate cropped files
        self.loop(number_subjects=number_subjects)


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description='Generates resampled files (either skeletons or foldlabels)')
    parser.add_argument(
        "-g", "--graph_dir", type=str, default=_GRAPH_DIR_DEFAULT,
        help='Source directory where the graph lies. '
             'Default is : ' + _GRAPH_DIR_DEFAULT)
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT,
        help='Source directory where inputs files (skeletons or labels) lie. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to store the cropped and pickle files. '
             'Default is : ' + _TGT_DIR_DEFAULT)
    parser.add_argument(
        "-m",
        "--morphologist_dir",
        type=str,
        default=_MORPHOLOGIST_DIR_DEFAULT,
        help='Directory where subjects to be processed are stored')
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        "-x",
        "--out_voxel_size",
        type=float,
        nargs='+',
        default=_OUT_VOXEL_SIZE,
        help='Voxel size of output images'
        'Default is : 1 1 1')
    parser.add_argument(
        "-v", "--verbose",
        default=False,
        action='store_true',
        help='If verbose is true, no parallelism.')

    params = {}

    args = parser.parse_args(argv)

    # Writes command line argument to target dir for logging
    log_command_line(args, "resample_skeletons.py", args.tgt_dir)

    params['src_dir'] = args.src_dir
    params['graph_dir'] = args.graph_dir
    params['tgt_dir'] = args.tgt_dir
    params['side'] = args.side
    params['out_voxel_size'] = tuple(args.out_voxel_size)
    params['morphologist_dir'] = args.morphologist_dir
    params['verbose'] = args.verbose

    number_subjects = args.nb_subjects

    # Check if nb_subjects is either the string "all" or a positive integer
    try:
        if number_subjects == "all":
            number_subjects = _ALL_SUBJECTS
        else:
            number_subjects = int(number_subjects)
            if number_subjects < 0:
                raise ValueError
    except ValueError:
        raise ValueError(
            "number_subjects must be either the string \"all\" or an integer")
    params['nb_subjects'] = number_subjects

    return params


def resample_skeletons(graph_dir=_GRAPH_DIR_DEFAULT,
                       src_dir=_SRC_DIR_DEFAULT,
                       tgt_dir=_TGT_DIR_DEFAULT,
                       morphologist_dir=_MORPHOLOGIST_DIR_DEFAULT,
                       side=_SIDE_DEFAULT,
                       list_sulci=_SULCUS_DEFAULT,
                       number_subjects=_ALL_SUBJECTS,
                       out_voxel_size=_OUT_VOXEL_SIZE,
                       verbose=_VERBOSE_DEFAULT):

    dataset = DatasetResampledSkeleton(graph_dir=graph_dir,
                                       src_dir=src_dir,
                                       tgt_dir=tgt_dir,
                                       morphologist_dir=morphologist_dir,
                                       side=side,
                                       out_voxel_size=out_voxel_size,
                                       verbose=verbose)
    dataset.loop(number_subjects=number_subjects)


def main(argv):
    """Reads argument line and creates cropped files and pickle file

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        params = parse_args(argv)

        # Actual API
        resample_skeletons(graph_dir=params['graph_dir'],
                           src_dir=params['src_dir'],
                           tgt_dir=params['tgt_dir'],
                           morphologist_dir=params['morphologist_dir'],
                           side=params['side'],
                           number_subjects=params['nb_subjects'],
                           out_voxel_size=params['out_voxel_size'],
                           verbose=params['verbose'])
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
