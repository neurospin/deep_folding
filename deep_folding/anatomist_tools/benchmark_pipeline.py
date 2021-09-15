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
from pqdm.processes import pqdm
from joblib import cpu_count

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
        "-t", "--tgt_dir", type=str, default=_TGT_DIR_DEFAULT,
        help='Target directory where to save the benchmark. '
             'Default is : ' + _TGT_DIR_DEFAULT)
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
    tgt_dir = args.tgt_dir  # src_dir is a string
    sulcus = args.sulcus  # sulcus is a list
    side = args.side
    ss_size = args.ss_size
    mode = args.benchmark_mode
    bench_size = args.benchmark_size
    resampling = args.resampling
    bbox_dir = args.bbox_dir
    subjects_list = args.subjects_list

    return tgt_dir, sulcus, side, ss_size, mode, bench_size, resampling, bbox_dir, subjects_list


_SS_SIZE_DEFAULT = 1000
_TGT_DIR_DEFAULT = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/'
_SULCUS_DEFAULT = ['S.T.s.ter.asc.ant.', 'S.T.s.ter.asc.post.']
_SIDE_DEFAULT = 'R'
_MODE_DEFAULT = 'suppress'
_BENCH_SIZE = 150
_RESAMPLING_DEFAULT = None
_BBOX_DIR_DEFAULT = '/neurospin/dico/data/deep_folding/data/bbox'
_SUBJECT_LIST_DEFAULT = None


def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus-2, 1)


class BenchmarkPipe:
    """
    """

    def __init__(self, tgt_dir, sulcus, side, ss_size, mode, bench_size,
                 resampling, bbox_dir, subjects_list):
        self.resampling = resampling
        self.mode = mode
        self.sulcus = complete_sulci_name(sulcus, side)
        self.ss_size = ss_size
        self.bench_size = bench_size
        self.bbox_dir = bbox_dir
        self.side = side
        self.subjects_list = subjects_list
        self.b_num = len(next(os.walk(tgt_dir))[1]) + 1
        print(self.b_num)
        self.tgt_dir = os.path.join(tgt_dir, 'benchmark'+str(self.b_num))
        if not os.path.isdir(self.tgt_dir):
            os.mkdir(self.tgt_dir)
        print(self.mode)

    def get_sub_list(self):
        list_subjects = []
        for img in os.listdir(self.tgt_dir):
            if '.nii.gz' in img and 'minf' not in img:
                sub = re.search('_(\d{6})', img).group(1)
                list_subjects.append(sub)
        return list_subjects


    def crop_one_file(self, sub):
        dir_m = '/neurospin/dico/lguillon/skeleton/transfo_pre_process/natif_to_template_spm_' + sub +'.trm'
        dir_r = '/neurospin/hcp/ANALYSIS/3T_morphologist/' + sub + '/t1mri/default_acquisition/normalized_SPM_' + sub +'.nii'
        skel_prefix = 'output_skeleton_'
        file_skeleton = os.path.join(self.tgt_dir, skel_prefix + sub + '.nii.gz')
        file_cropped = os.path.join(self.tgt_dir, sub + '_normalized.nii.gz')

        if self.resampling:
            resampled = resample(file_skeleton, output_vs=(2, 2, 2),
                                 transformation=dir_m)

            aims.write(resampled, file_cropped)
        else:
            cmd_normalize = "AimsApplyTransform -i " + file_skeleton + \
                            " -o " + file_cropped +" -m " + dir_m + " -r " + \
                            dir_r + " -t nearest"
            os.system(cmd_normalize)

        # Crop of the images
        if self.mode == 'random':
            # 42 instead of 0 in order to avoid crops with only black voxels
            # Int 108 and 91 depend on downsampling and normalization
            random_x = random.randint(0, 42-self.box_size[0]-1)
            random_y = random.randint(0, 108-self.box_size[1]-1)
            random_z = random.randint(0, 91-self.box_size[2]-1)
            xmax, ymax, zmax = random_x + self.box_size[0], random_y + self.box_size[1], random_z + self.box_size[2]
            cmd_bounding_box = ' -x ' + str(random_x) + ' -y ' + str(random_y) + \
                               ' -z ' + str(random_z) + ' -X '+ str(xmax) + ' -Y ' + str(ymax) + ' -Z ' + str(zmax)

        else:
            if self.mode == 'asymmetry':
                # for left hemisphere
                xmin, ymin, zmin = '52', '50', '12'
                xmax, ymax, zmax = '74', '86', '47'

            cmd_bounding_box = ' -x ' + xmin + ' -y ' + ymin + ' -z ' + zmin + ' -X '+ xmax + ' -Y ' + ymax + ' -Z ' + zmax

        cmd_crop = "AimsSubVolume -i " + file_cropped + " -o " + file_cropped + cmd_bounding_box
        os.system(cmd_crop)

        if self.mode == 'asymmetry':
            cmd_flip = "AimsFlip -i " + file_cropped + " -o " + file_cropped + " -m XX"
            os.system(cmd_flip)

    def launch_pipe(self):
        print(' ')
        print('Sulci list: ', self.sulcus)
        print('Mode chosen:', self.mode)
        print('Chosen Benchmark size: ', self.bench_size)
        print(' ')

        print('=================== Selection and possible alteration of benchmark skeletons ===================')
        generate(self.b_num, self.side, self.ss_size, sulci_list=self.sulcus,
                 saving_dir=self.tgt_dir, mode=self.mode, bbox_dir=self.bbox_dir,
                 bench_size=self.bench_size, subjects_list=self.subjects_list
                )

        bbox = compute_max_box(self.sulcus, self.side, src_dir=self.bbox_dir)
        print(bbox)

        xmin, ymin, zmin = str(bbox[0][0]), str(bbox[0][1]), str(bbox[0][2])
        xmax, ymax, zmax = str(bbox[1][0]), str(bbox[1][1]), str(bbox[1][2])
        self.box_size = [int(xmax)-int(xmin), int(ymax)-int(ymin), int(zmax)-int(zmin)]

        print(' ')
        print('=================== Normalization and crop of skeletons ==================')

        list_subjects = self.get_sub_list()
        pqdm(list_subjects, self.crop_one_file, n_jobs=define_njobs())



def main(argv):
    tgt_dir, sulcus, side, ss_size, mode, bench_size, resampling, bbox_dir, subjects_list = parse_args(argv)


    benchmark = BenchmarkPipe(tgt_dir, sulcus, side, ss_size, mode, bench_size,
                              resampling, bbox_dir, subjects_list)

    benchmark.launch_pipe()


    input_dict = {'sulci_list': sulcus, 'simple_surface_min_size': ss_size,
                  'side': side, 'mode': mode}
    log_file = open(tgt_dir + "/logs.json", "a+")
    log_file.write(json.dumps(input_dict))
    log_file.close()


if __name__ == '__main__':
    main(argv=sys.argv[1:])
