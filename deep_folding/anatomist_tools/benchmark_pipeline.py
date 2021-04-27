#!/usr/bin/env python
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


"""Generating benchmark of sulcal abnormalities up tp pickle file

The aim of this script is to generate a benchmark of sulcal abnormalities.
The script gathers all the steps necessary.
"""

######################################################################
# Imports and global variables definitions
######################################################################

import os
import argparse
import sys
import re
import json
import benchmark_generation
import utils.load_bbox


def parse_args(argv):
    """Function parsing command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        src_dir: a list with source directory names, full path
        sulcus: a string containing the sulcus to analyze
        number_subjects: number of subjects to analyze
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='benchmark_pipeline.py',
        description='Generates benchmark of sulcal abnormalities')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_SRC_DIR_DEFAULT, nargs='+',
        help='Source directory where the MRI data lies. '
             'If there are several directories, add all directories '
             'one after the other. Example: -s DIR_1 DIR_2. '
             'Default is : ' + _SRC_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", default=_SULCUS_DEFAULT, nargs='+',
        help='Sulcus name around which we determine the bounding box. '
             'Default is : ' + str(_SULCUS_DEFAULT))
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side. Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-l", "--ss_size", type=int, default=_SS_SIZE_DEFAULT,
        help='simple surface min size Default is : ' + str(_SS_SIZE_DEFAULT))
    parser.add_argument(
        "-m", "--benchmark_mode", type=str, default=_MODE_DEFAULT,
        help='benchmark creation mode Default is : ' + str(_MODE_DEFAULT))

    args = parser.parse_args(argv)
    src_dir = args.src_dir  # src_dir is a list
    sulcus = args.sulcus  # sulcus is a string
    side = args.side
    ss_size = args.ss_size
    mode = args.benchmark_mode

    return src_dir, sulcus, side, ss_size, mode



_SS_SIZE_DEFAULT = 500
_SRC_DIR_DEFAULT = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/'
_SULCUS_DEFAULT = ['S.T.s.ter.asc.ant._left', 'S.T.s.ter.asc.post._left']
_SIDE_DEFAULT = 'L'
_MODE_DEFAULT = 'suppression'


def main(argv):
    """Main loop to generate abnormal skeletons and create pickle files
    """
    src_dir, sulcus, side, ss_size, mode = parse_args(argv)
    print(sulcus)
    if mode == 'suppression':
        b_num = len(os.walk(_SRC_DIR_DEFAULT).next()[1]) + 1
        tgt_dir = src_dir + 'benchmark' + str(b_num) + '/'
        if not os.path.isdir(tgt_dir):
            os.mkdir(tgt_dir)
        benchmark_generation.generate(b_num, side, ss_size, sulcus)
    elif mode=='merge':
        src_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark_merge/'
        b_num = len(os.walk(src_dir).next()[1]) + 1
        tgt_dir = src_dir + 'benchmark' + str(b_num) + '/'
        if not os.path.isdir(tgt_dir):
            os.mkdir(tgt_dir)
        benchmark_generation.generate_add_ss(b_num, side, ss_size, sulcus)

    # Get bounding box in voxels
    bbox = utils.load_bbox.compute_max_box(sulcus, side)
    print(bbox)
    # Take the coordinates of the bounding box
    xmin, ymin, zmin = str(bbox[0][0]), str(bbox[0][1]), str(bbox[0][2])
    xmax, ymax, zmax = str(bbox[1][0]), str(bbox[1][1]), str(bbox[1][2])

    for img in os.listdir(tgt_dir):
        if '.nii.gz' in img and 'minf' not in img:
            sub = re.search('_(\d{6})', img).group(1)

            # Normalization and resampling of altered skeleton images
            dir_m = '/neurospin/dico/lguillon/skeleton/transfo_pre_process/natif_to_template_spm_' + sub +'.trm'
            dir_r = '/neurospin/hcp/ANALYSIS/3T_morphologist/' + sub + '/t1mri/default_acquisition/normalized_SPM_' + sub +'.nii'
            cmd_normalize = "AimsResample -i " + tgt_dir + img + " -o " + tgt_dir + img[:-7] + "_normalized.nii.gz -m " + dir_m + " -r " + dir_r
            os.system(cmd_normalize)

            # Crop of the images
            file = tgt_dir + img[:-7] + "_normalized.nii.gz"
            cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + ' -X '+ xmax + ' -Y ' + ymax + ' -Z ' + zmax
            cmd_crop = "AimsSubVolume -i " + file + " -o " + file + cmd_bounding_box
            os.system(cmd_crop)

    input_dict = {'sulci_list': sulcus, 'simple_surface_min_size': ss_size,
                  'side': side}
    log_file = open(tgt_dir + "logs.json", "a+")
    log_file.write(json.dumps(input_dict))
    log_file.close()


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
