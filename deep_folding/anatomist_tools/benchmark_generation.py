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


""" Creating abnormal skeleton images

The aim of this script is to generate a benchmark of sulcal abnormalities.
Abnormalities are defined as skeletons with one simple surface missing.
This simple surface must be completely inside the bounding box of interest and
include a minimum number of voxels (in order that the anomaly is big enough to
be considered as abnormal).
"""

######################################################################
# Imports and global variables definitions
######################################################################

from soma import aims
import numpy as np
from glob import glob
import random
import pandas as pd
import os
import utils.load_bbox


class Benchmark():
    def __init__(self, b_num, side, ss_size, sulci_list,
                 data_dir=_DEFAULT_DATA_DIR, saving_dir=_DEFAULT_SAVING_DIR):
        self.b_num = b_num
        self.side = side
        self.ss_size = ss_size
        self.sulci_list = sulci_list
        self.data_dir = data_dir
        self.saving_dir = saving_dir
        self.subjects_list = get_sub_list()
        self.abnormality_test = []
        self.bbmin, self.bbmax = utils.load_bbox.compute_max_box(sulci_list, side, talairach_box=True)
        print(self.bbmin, self.bbmax)

    def get_one_simple_surface(self, sub):

        cpt_arg_1 = '/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/'
        cpt_arg_2 = '_default_session_auto.arg'
        cpt_skel_1 = '/t1mri/default_acquisition/default_analysis/segmentation/'
        cpt_skel_2 = 'skeleton_'
        cpt_skel_3 = '.nii.gz'

        if os.path.isdir(self.data_dir + str(sub)):
            surfaces = dict()

            graph = aims.read(os.path.join(self.data_dir, str(sub),
                              cpt_arg_1, side, str(sub), cpt_arg_2))
            self.skel = aims.read(os.path.join(self.data_dir, str(sub), cpt_skel_1,
                            side, cpt_skel_2, str(sub), cpt_skel_3))

            for v in graph.vertices():
                if 'label' in v:
                    bbmin_surface = v['Tal_boundingbox_min']
                    bbmax_surface = v['Tal_boundingbox_max']
                    bck_map = v['aims_ss']

                    if all([a >= b for (a, b) in zip(bbmin_surface, self.bbmin)]) and all([a <= b for (a, b) in zip(bbmax_surface, self.bbmax)]):
                        for bucket in bck_map:
                            if bucket.size() > self.ss_size: # In order to keep only large enough simple surfaces
                                surfaces[len(surfaces)] = v

            print(len(surfaces.keys()))
        return surfaces


    def alter_skeleton(self):

        # Suppression of one random simple surface (satisfying both criteria)
        surface = random.randint(0, len(surfaces)-1)
        print(surfaces[surface]['label'])

        bck_map = surfaces[surface]['aims_ss']
        for voxel in bck_map[0].keys():
            self.skel.setValue(0, voxel[0], voxel[1], voxel[2])

        bck_map_bottom = surfaces[surface]['aims_bottom']
        for voxel in bck_map_bottom[0].keys():
            self.skel.setValue(0, voxel[0], voxel[1], voxel[2])

        return skel


    def save_file(self):
        # Writting of output graph
        fileout = self.saving_dir + 'output_skeleton_' + str(sub) + '.nii.gz'
        print('writing altered skeleton to', fileout)
        aims.write(self.skel, fileout)


    def save_lists(self, abnormality_test):
        nor_set = list(set(subjects_list) - set(abnormality_test))

        print('Dataset split, train + val + normal test: ', len(nor_set),
              'abnormal test: ', len(abnormality_test))

        df_train = pd.DataFrame(nor_set)
        df_train.to_csv(saving_dir + 'train.csv')

        df_abnor_test = pd.DataFrame(abnormality_test)
        df_abnor_test.to_csv(saving_dir + 'abnormality_test.csv')



def get_sub_list():
    """Returns subjects list for which latered skelerons can be created
    Only right handed HCP subjects
    """
    right_handed = pd.read_csv('/neurospin/dico/lguillon/hcp_info/right_handed.csv')
    subjects_list = list(right_handed['Subject'])
    random.shuffle(subjects_list)

    return subjects_list


def generate(b_num, side, ss_size, sulci_list, mode='suppress'):
    b_num =111
    benchmark = Benchmark(b_num, side, ss_size, sulci_list)
    abnormality_test = []
    for sub in subjects_list:
        print(sub)
        surfaces = benchmark.get_one_simple_surface(sub)
        if len(surfaces.keys()) > 0:
            benchmark.alter_skeleton(surfaces)
            # Addition of modified graph to abnormality_test set
            abnormality_test.append(sub)
            if len(abnormality_test) == 150:
                break

        if test==False:
            benchmark.save_file()
            benchmark.save_lists(abnormality_test)




"""
def generate(b_num, side, ss_size, sulci_list):
    Generates the abnormal skeleton images

    Args: b_num: the benchmark number to create (int)
          side: hemisphere, str, whether 'L' or 'R'
          ss_size: Minimal size of simple surface to suppress
          sulci_list: the list of sulci delimiting the bounding box

    Returns: list of subjects altered or original
             saves abnormal skeletons to saving_dir

    # folder containing all HCP subjects folder
    data_dir = '/neurospin/hcp/ANALYSIS/3T_morphologist/'
    saving_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark' + str(b_num) +'/'

    # List of right handed subjects
    subjects_list = get_sub_list()

    # Saving of simple surfaces satisfying both criteria: completely inside the
    # bounding box defined for the crop and including at least ss_size voxels
    abnormality_test = []
    bbmin, bbmax = utils.load_bbox.compute_max_box(sulci_list, side, talairach_box=True)
    print(bbmin, bbmax)

    for sub in subjects_list:
        print(sub)
        if os.path.isdir(data_dir + str(sub)):
            surfaces = dict()

            graph = aims.read(data_dir + str(sub) + '/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/'+ side + str(sub) + '_default_session_auto.arg')
            skel = aims.read(data_dir + str(sub) + '/t1mri/default_acquisition/default_analysis/segmentation/' + side + 'skeleton_' + str(sub) + '.nii.gz')

            for v in graph.vertices():
                if 'label' in v:
                    bbmin_surface = v['Tal_boundingbox_min']
                    bbmax_surface = v['Tal_boundingbox_max']
                    bck_map = v['aims_ss']

                    if all([a >= b for (a, b) in zip(bbmin_surface, bbmin)]) and all([a <= b for (a, b) in zip(bbmax_surface, bbmax)]):
                        for bucket in bck_map:
                            if bucket.size() > ss_size: # In order to keep only large enough simple surfaces
                                surfaces[len(surfaces)] = v

            print(len(surfaces.keys()))
            if len(surfaces.keys()) > 0:
                # Suppression of one random simple surface (satisfying both criteria)
                surface = random.randint(0, len(surfaces)-1)
                print(surfaces[surface]['label'])

                bck_map = surfaces[surface]['aims_ss']
                for voxel in bck_map[0].keys():
                    skel.setValue(0, voxel[0], voxel[1], voxel[2])

                bck_map_bottom = surfaces[surface]['aims_bottom']
                for voxel in bck_map_bottom[0].keys():
                    skel.setValue(0, voxel[0], voxel[1], voxel[2])

                # Writting of output graph
                fileout = saving_dir + 'output_skeleton_' + str(sub) + '.nii.gz'
                print('writing altered skeleton to', fileout)
                aims.write(skel, fileout)

                # Addition of modified graph to abnormality_test set
                abnormality_test.append(sub)
                if len(abnormality_test) == 150:
                    break

    # Train, validation and test separation
    # It is important that examples of this benchmark aren't used during training and
    # validation phase. In order to make sure this is the case, 2 lists are created.
    nor_set = list(set(subjects_list) - set(abnormality_test))

    print('Dataset split, train + val + normal test: ', len(nor_set),
          'abnormal test: ', len(abnormality_test))

    df_train = pd.DataFrame(nor_set)
    df_train.to_csv(saving_dir + 'train.csv')

    df_abnor_test = pd.DataFrame(abnormality_test)
    df_abnor_test.to_csv(saving_dir + 'abnormality_test.csv')"""


def generate_add_ss(b_num, side, ss_size, sulci_list):
    """
    Creates a benchmark with skeletons having additional simple surfaces (ss)
    IN: b_num: benchmark number
        side: hemisphere, str, whether 'L' or 'R',
        region: region of the brain
        ss_size: Minimal size of simple surface to suppress
    OUT: altered skeletons generated
         list of subjects altered or original
    """
    n = 150 # benchmark size
    data_dir = '/neurospin/hcp/ANALYSIS/3T_morphologist/' # folder containing all HCP subjects folder
    saving_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark_merge/benchmark' + str(b_num) + '/'

    subjects_list = get_sub_list()

    abnormality_test, givers = [], []
    bbmin, bbmax = utils.load_bbox.compute_max_box(sulci_list, side, talairach_box=True)
    print(bbmin, bbmax)

    for i, sub in enumerate(subjects_list):
        print(sub)
        if os.path.isdir(data_dir + str(sub)):
            surfaces = dict()
            graph = aims.read(data_dir + str(sub) + '/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/'+ side + str(sub) + '_default_session_auto.arg')

            for v in graph.vertices():
                if 'label' in v:
                    bbmin_surface = v['Tal_boundingbox_min']
                    bbmax_surface = v['Tal_boundingbox_max']
                    bck_map = v['aims_ss']

                    if all([a >= b for (a, b) in zip(bbmin_surface, bbmin)]) and all([a <= b for (a, b) in zip(bbmax_surface, bbmax)]):
                        for bucket in bck_map:
                            if bucket.size() > ss_size: # In order to keep only large enough simple surfaces
                                surfaces[len(surfaces)] = v

            if len(surfaces.keys())>0: # there is at least one SS that satisfies conditions
                surface = random.randint(0, len(surfaces)-1)
                print(surfaces[surface]['label'])

                # Loading of skeleton on which SS will be added
                sub_added = subjects_list[i+1]
                if os.path.isdir(data_dir + str(sub_added)):
                    skel = aims.read(data_dir + str(sub_added) + '/t1mri/default_acquisition/default_analysis/segmentation/' + side + 'skeleton_' + str(sub_added) + '.nii.gz')
                    bck_map = surfaces[surface]['aims_ss']
                    for voxel in bck_map[0].keys():
                        if skel.value(voxel[0], voxel[1], voxel[2])!=11:
                            skel.setValue(60, voxel[0], voxel[1], voxel[2])

                    bck_map_bottom = surfaces[surface]['aims_bottom']
                    for voxel in bck_map_bottom[0].keys():
                        if skel.value(voxel[0], voxel[1], voxel[2])!=11:
                            skel.setValue(60, voxel[0], voxel[1], voxel[2])

                    # Writting of output graph
                    fileout = saving_dir + 'output_skeleton_test_' + str(sub_added) + '.nii.gz'
                    print('writing altered skeleton to', fileout)
                    aims.write(skel, fileout)

                    # Addition of modified graph to abnormality_test set
                    abnormality_test.append(sub_added)
                    givers.append(sub)
                    if len(abnormality_test) == n:
                        break

    # Train, validation and test separation
    # It is important that examples of this benchmark aren't used during training and
    # validation phase. In order to make sure this is the case, 2 lists are created.
    nor_set = list(set(subjects_list) - set(abnormality_test))
    nor_set = list(set(nor_set)- set(givers))

    print('Dataset split, train + normal test: ', len(nor_set),
          'abnormal test: ', len(abnormality_test),
          'givers: ', len(givers))

    df_train = pd.DataFrame(nor_set)
    df_train.to_csv(saving_dir + 'train.csv')

    df_abnor_test = pd.DataFrame(abnormality_test)
    df_abnor_test.to_csv(saving_dir + 'abnormality_test.csv')

    df_givers = pd.DataFrame(givers)
    df_givers.to_csv(saving_dir + 'givers.csv')

"""
if __name__ == '__main__':
    b_num = 1
    side = 'L'
    sulci_list = ['S.T.s.ter.asc.ant._left', 'S.T.s.ter.asc.post._left']
    ss_size = 500
    generate_add_ss(b_num, side, ss_size, sulci_list)"""
