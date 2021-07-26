# -*- coding: utf-8 -*-
# /usr/bin/env python2.7 + brainvisa compliant env
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

""" Full pipeline to create a benchmark of abnormalities.

"""

######################################################################
# Imports and global variables definitions
######################################################################
from __future__ import division
from __future__ import print_function

from benchmark_generation import *
from deep_folding.anatomist_tools.utils.resample import resample
from deep_folding.anatomist_tools.utils.sulcus_side import complete_sulci_name

import re
import sys
import argparse
import json


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
        prog='create_benchmark.py',
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
    parser.add_argument(
        "-b", "--benchmark_size", type=int, default=_BENCH_SIZE,
        help='benchmark size Default is : ' + str(_BENCH_SIZE))
    parser.add_argument(
        "-p", "--resampling", type=str, default=None,
        help='Method of resampling to perform. '
             'Type of resampling: s[ulcus] for Bastien method'
             'If None, AimsApplyTransform is used.'
             'Default is : None')
    parser.add_argument(
        "-o", "--bbox_dir", type=str, default=_BBOX_DIR_DEFAULT,
        help="Bounding box directory where json files containing "
             "bounding box coordinates have been stored. "
             "Default is : " + _BBOX_DIR_DEFAULT)
    parser.add_argument(
        "-j", "--subjects_list", type=str, default=_SUBJECT_LIST_DEFAULT,
        help="Subjects list from which create benchmark "
             "Default is : " + str(_SUBJECT_LIST_DEFAULT))

    args = parser.parse_args(argv)
    src_dir = args.src_dir  # src_dir is a list
    sulcus = args.sulcus  # sulcus is a string
    side = args.side
    ss_size = args.ss_size
    mode = args.benchmark_mode
    bench_size = args.benchmark_size
    resampling = args.resampling
    bbox_dir = args.bbox_dir
    subjects_list = args.subjects_list

    return src_dir, sulcus, side, ss_size, mode, bench_size, resampling, bbox_dir, subjects_list


_SS_SIZE_DEFAULT = 1000
_SRC_DIR_DEFAULT = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/'
_SULCUS_DEFAULT = ['S.T.s.ter.asc.ant.', 'S.T.s.ter.asc.post.']
_SIDE_DEFAULT = 'R'
_MODE_DEFAULT = 'suppress'
_BENCH_SIZE = 150
_RESAMPLING_DEFAULT = None
_BBOX_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/data/bbox'
_SUBJECT_LIST_DEFAULT = None

def main(argv):
    src_dir, sulcus, side, ss_size, mode, bench_size, resampling, bbox_dir, subjects_list = parse_args(argv)
    sulcus = complete_sulci_name(sulcus, side)
    b_num = len(next(os.walk(src_dir))[1]) + 1
    tgt_dir = os.path.join(src_dir, 'benchmark'+str(b_num))
    if not os.path.isdir(tgt_dir):
        os.mkdir(tgt_dir)

    print(' ')
    print('Sulci list: ', sulcus)
    print('Mode chosen:', mode)
    print('Chosen Benchmark size: ', bench_size)
    print(' ')

    print('=================== Selection and possible alteration of benchmark skeletons ===================')
    generate(b_num, side, ss_size, sulci_list=sulcus,
             mode=mode, bench_size=bench_size, subjects_list=subjects_list)

    bbox = compute_max_box(sulcus, side, src_dir=bbox_dir)
    print(bbox)

    xmin, ymin, zmin = str(bbox[0][0]), str(bbox[0][1]), str(bbox[0][2])
    xmax, ymax, zmax = str(bbox[1][0]), str(bbox[1][1]), str(bbox[1][2])
    box_size = [int(xmax)-int(xmin), int(ymax)-int(ymin), int(zmax)-int(zmin)]

    print(' ')
    print('=================== Normalization and crop of skeletons ==================')
    for img in os.listdir(tgt_dir):
        if '.nii.gz' in img and 'minf' not in img:
            sub = re.search('_(\d{6})', img).group(1)
            # Normalization and resampling of altered skeleton images
            dir_m = '/neurospin/dico/lguillon/skeleton/transfo_pre_process/natif_to_template_spm_' + sub +'.trm'
            dir_r = '/neurospin/hcp/ANALYSIS/3T_morphologist/' + sub + '/t1mri/default_acquisition/normalized_SPM_' + sub +'.nii'
            file_skeleton = tgt_dir + '/' + img
            file_cropped = tgt_dir + '/' + img[:-7] + "_normalized.nii.gz"

            if resampling:
                resample(file_skeleton, file_cropped, output_vs=(2, 2, 2),
                         transformation=dir_m)
            else:
                cmd_normalize = "AimsApplyTransform -i " + tgt_dir +'/' + img + \
                                " -o " + tgt_dir + '/' + img[:-7] + \
                                "_normalized.nii.gz -m " + dir_m + " -r " + \
                                dir_r + " -t nearest"
                os.system(cmd_normalize)

            # Crop of the images
            file = os.path.join(tgt_dir, img[:-7] + '_normalized.nii.gz')
            if mode == 'random':
                # 40 instead of 0 in order to avoid crops with only black voxels
                random_x = random.randint(40, 157-box_size[0]-1)
                random_y = random.randint(0, 189-box_size[1]-1)
                random_z = random.randint(0, 136-box_size[2]-1)
                print(random_x, random_y, random_z)
                xmax, ymax, zmax = random_x + box_size[0], random_y + box_size[1], random_z + box_size[2]
                cmd_bounding_box = ' -x ' + str(random_x) + ' -y ' + str(random_y) + \
                                   ' -z ' + str(random_z) + ' -X '+ str(xmax) + ' -Y ' + str(ymax) + ' -Z ' + str(zmax)
                print(cmd_bounding_box)

            else:
                if mode == 'asymmetry':
                    # for left hemisphere
                    xmin, ymin, zmin = '52', '50', '12'
                    xmax, ymax, zmax = '74', '86', '47'

                cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + ' -X '+ xmax + ' -Y ' + ymax + ' -Z ' + zmax

            cmd_crop = "AimsSubVolume -i " + file + " -o " + file + cmd_bounding_box
            os.system(cmd_crop)

            if mode == 'asymmetry':
                print('ici')
                cmd_flip = "AimsFlip -i " + file + " -o " + file + " -m XX"
                os.system(cmd_flip)

    input_dict = {'sulci_list': sulcus, 'simple_surface_min_size': ss_size,
                  'side': side, 'mode': mode}
    log_file = open(tgt_dir + "/logs.json", "a+")
    log_file.write(json.dumps(input_dict))
    log_file.close()


if __name__ == '__main__':
    main(argv=sys.argv[1:])
