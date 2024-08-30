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

"""Creating npy file from T1 MRI datas
The aim of this script is to create dataset of distances to bottoms
alogn skeletons saved in a .npy file.

Crops of skeletons must be generated before
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
import pandas as pd

from os.path import join
from os.path import basename
from p_tqdm import p_map

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.save_data import save_to_numpy
from deep_folding.brainvisa.utils.save_data import \
    save_to_dataframe_format_from_list
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import LogJson
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.distbottom import generate_distbottom
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_distbottom_files, \
    compare_number_aims_files_with_number_in_source, \
    save_list_to_csv
from deep_folding.config.logs import set_file_logger


# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, \
    _CROP_DIR_DEFAULT, \
    _SIDE_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def quality_checks(crop_dir, side):
    s = np.load(f"{crop_dir}/{side}skeleton.npy")
    d = np.load(f"{crop_dir}/{side}distbottom.npy")

    # checks if same voxel position
    assert (s.shape == d.shape), (
        f"Skeleton and distbottom of different shapes: {s.shape} != {d.shape}")
    assert (s[d == 32501].sum() == 0), (
        f"Skeleton and distbottom with different non-zero positions: "
        f"{(s[d == 32501]!=0).sum()} different non-zero positions")
    assert ((d[s == 0] != 32501).sum() == 0), (
        f"Skeleton and distbottom with different non-zero positions: "
        f"{(d[s == 0] != 32501).sum()} different non-zero positions")

    # Checks if subjects are equal between distbottom and skeleton
    dfs = pd.read_csv(f"{crop_dir}/{side}skeleton_subject.csv")
    dfd = pd.read_csv(f"{crop_dir}/{side}distbottom_subject.csv")
    assert (dfs.equals(dfd)), \
        "List of subjects for distbottom and skeleton are not equal"

    # Checks if numpy arrays and csvs are consistent
    assert (s.shape[0] == len(dfs)), \
        "Number of skeleton subjects differs between numpy array and csv"
    assert (d.shape[0] == len(dfd)), \
        "Number of distbottom subjects differs between numpy array and csv"


class DistBottomCropGenerator:
    """Generates cropped skeleton files and corresponding npy file
    """

    def __init__(self,
                 src_dir=_CROP_DIR_DEFAULT,
                 crop_dir=_CROP_DIR_DEFAULT,
                 side=_SIDE_DEFAULT,
                 parallel=False):
        """Inits with list of directories
        Args:
            src_dir: folder containing generated skeletons, labels or distmaps
            crop_dir: name of output directory for crops with full path
            side: hemisphere side (either L for left,
                                   or R for right hemisphere)
        """

        self.crop_dir = crop_dir
        self.side = side
        self.parallel = parallel

        # Names of files in function of dictionary:
        #               keys -> 'subject' and 'side'
        # Generated skeleton from folding graphs
        self.src_dir = join(src_dir, f"{self.side}crops")

        # Directory where to store cropped distbottom files
        self.cropped_samples_dir = join(
            self.crop_dir, self.side + 'distbottom')

        # Names of files in function of dict: keys -> 'subject' and 'side'
        # Generated skeleton crops
        self.src_file = join(
            self.src_dir,
            '%(subject)s_cropped_skeleton.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.cropped_file = '%(subject)s_cropped_distbottom.nii.gz'

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^(.*)_cropped_skeleton.nii.gz$'

        # Creates json log class
        json_file = join(self.crop_dir, self.side + 'distbottom.json')
        self.json = LogJson(json_file)

        # Creates npys file name
        self.file_basename_npy = self.side + 'distbottom'
        self.file_basename_pickle = self.side + 'distbottom'

    def generate_one_distbottom(self, subject_id):
        """Crops one file
        Args:
            subject_id: string giving the subject ID
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        # subject = {'subject': subject_id, 'side': self.side}
        # FOR TISSIER
        # subject_id = re.search('([ae\d]{5,6})', subject_id).group(0)

        # Skeleton file name
        file_src = self.src_file % {'subject': subject_id}

        if os.path.exists(file_src):
            # Creates output (cropped) file name
            file_cropped = join(
                self.cropped_samples_dir,
                self.cropped_file % {
                    'subject': subject_id})

            generate_distbottom(file_src, file_cropped)

        else:
            raise FileNotFoundError(f"{file_src} not found")

    def generate_distbottom_files(self, nb_subjects=_ALL_SUBJECTS):
        """Generate distbottom files
        The programm loops over all subjects from the input (source) directory.
        Args:
            nb_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if nb_subjects:

            if os.path.isdir(self.src_dir):
                files = glob.glob(f"{self.src_dir}/*.nii.gz")
                log.debug(f"Nifti files in {self.src_dir} = {files}")
                log.debug(f"Regular expresson is: {self.expr}")

                # Creates target directories
                create_folder(self.crop_dir)
                create_folder(self.cropped_samples_dir)

                # Generates list of subjects not treated yet
                not_processed_files = get_not_processed_distbottom_files(
                    self.src_dir,
                    self.cropped_samples_dir)

                if len(files):
                    list_all_subjects = [
                        re.search(self.expr, basename(dI))[1]
                        for dI in not_processed_files]
                else:
                    raise ValueError(f"no nifti files in {self.src_dir}")
            else:
                raise NotADirectoryError(
                    f"{self.src_dir} doesn't exist or is not a directory")

            if len(list_all_subjects):
                # Gives the possibility to list
                # only the first nb_subjects
                list_subjects = select_subjects_int(list_all_subjects,
                                                    nb_subjects)

                log.info(f"Expected number of subjects = {len(list_subjects)}")
                log.info(f"list_subjects[:5] = {list_subjects[:5]}")
                log.debug(f"list_subjects = {list_subjects}")

                # Creates target and cropped directory
                create_folder(self.crop_dir)
                create_folder(self.cropped_samples_dir)

                # Writes number of subjects and directory names to json file
                dict_to_add = {
                    'nb_subjects': len(list_subjects),
                    'src_dir': self.src_dir,
                    'side': self.side,
                    'crop_dir': self.crop_dir}
                self.json.update(dict_to_add=dict_to_add)

                if self.parallel:
                    log.info(
                        "PARALLEL MODE: subjects are in parallel")
                    p_map(
                        self.generate_one_distbottom,
                        list_subjects,
                        num_cpus=define_njobs())
                else:
                    log.info(
                        "SERIAL MODE: subjects are scanned serially")
                    for sub in list_subjects:
                        self.generate_one_distbottom(sub)
            else:
                list_subjects = []
                log.info(
                    "There is no subject or there is no subject to process"
                    "in the source directory")

            # Checks if there is expected number of generated files
            compare_number_aims_files_with_expected(self.cropped_samples_dir,
                                                    list_subjects)

            # Checks if number of generated files == number of src files
            crop_files, src_files = \
                compare_number_aims_files_with_number_in_source(
                    self.cropped_samples_dir,
                    self.src_dir)
            not_processed_files = get_not_processed_distbottom_files(
                self.src_dir, self.cropped_samples_dir)
            save_list_to_csv(not_processed_files,
                             f"{self.crop_dir}/not_processed_files.csv")

    def compute(self, nb_subjects=_ALL_SUBJECTS):
        """Main API to create numpy files
        The programm loops over all subjects from the input (source) directory.
        Args:
            nb_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        self.json.write_general_info()

        # Generate cropped files
        self.generate_distbottom_files(nb_subjects=nb_subjects)

        # Creation of .npy file containing all subjects
        if nb_subjects:
            list_sample_id, list_sample_file = \
                save_to_numpy(cropped_dir=self.cropped_samples_dir,
                              tgt_dir=self.crop_dir,
                              file_basename=self.file_basename_npy,
                              parallel=self.parallel)
            save_to_dataframe_format_from_list(
                cropped_dir=self.cropped_samples_dir,
                tgt_dir=self.crop_dir,
                file_basename=self.file_basename_pickle,
                list_sample_id=list_sample_id,
                list_sample_file=list_sample_file)

        quality_checks(self.crop_dir, self.side)


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
        description='Generates cropped and npy files of distances to bottom')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_CROP_DIR_DEFAULT,
        help='Source directory where cropped skeleton input files lie. '
             'Default is : ' + _CROP_DIR_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_CROP_DIR_DEFAULT,
        help='Output directory where to store the cropped distbottom files. '
             'Default is : ' + _CROP_DIR_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
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

    params = {}

    args = parser.parse_args(argv)

    # Writes command line argument to target dir for logging
    setup_log(
        args,
        log_dir=f"{args.output_dir}",
        prog_name=basename(__file__),
        suffix="right" if args.side == 'R' else 'left')

    params = vars(args)

    params['crop_dir'] = args.output_dir
    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    # Removes renamed params
    # So that we can use params dictionary directly as function arguments
    params.pop('output_dir')
    params.pop('verbose')

    return params


def generate_distbottom_crops(
        src_dir=_CROP_DIR_DEFAULT,
        crop_dir=_CROP_DIR_DEFAULT,
        side=_SIDE_DEFAULT,
        nb_subjects=_ALL_SUBJECTS,
        parallel=False
        ):

    # Gets function arguments and values
    params = locals()
    nb_subjects = params.pop('nb_subjects')

    # Initialization with same arguments and values as function
    crop = DistBottomCropGenerator(**params)

    # Actual generation of distbottom crops from graphs
    crop.compute(nb_subjects=nb_subjects)


@exception_handler
def main(argv):
    """Reads argument line and creates cropped files and npy file
    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    generate_distbottom_crops(**params)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
