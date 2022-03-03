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
from deep_folding.brainvisa.utils.bbox import compute_max_box
from deep_folding.brainvisa.utils.sulcus_side import complete_sulci_name


_DEFAULT_DATA_DIR = '/neurospin/hcp/ANALYSIS/3T_morphologist/'
_DEFAULT_SAVING_DIR = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/'
_DEFAULT_BBOX_DIR = '/neurospin/dico/data/deep_folding/data/bbox/'


class Benchmark():
    """Generates benchmark of altered skeletons
    """

    def __init__(self, b_num, side, ss_size, sulci_list, saving_dir,
                 data_dir=_DEFAULT_DATA_DIR, bbox_dir=_DEFAULT_BBOX_DIR):
        """Inits with list of directories, bounding box and sulci

        Args:
            b_num: number of benchmark
            side: hemisphere side (either L for left, or R for right hemisphere)
            ss_size: minimum size of simple surface to consider
            sulci_list: list of sulcus names
            data_dir: string naming full path source directories containing
                      MRI images
            saving_dir: name of directory where altered skeletons will be saved
        """
        self.b_num = b_num
        self.side = side
        self.ss_size = ss_size
        self.sulci_list = sulci_list
        self.sulci_list = complete_sulci_name(self.sulci_list, self.side)
        self.data_dir = data_dir
        self.saving_dir = saving_dir
        self.abnormality_test = []
        self.bbmin, self.bbmax = compute_max_box(
            self.sulci_list, side, talairach_box=True, src_dir=bbox_dir)
        print(self.bbmin, self.bbmax)
        self.cpt_skel_1 = 't1mri/default_acquisition/default_analysis/segmentation'
        self.cpt_skel_2 = 'skeleton_'
        self.cpt_skel_3 = '.nii.gz'

    def get_simple_surfaces(self, sub):
        """Selects simple surfaces of one subject that satisfy following
           conditions: size >= given ss_size and completely included in the
           bouding box

        Args:
            sub: int giving the subject
        """
        cpt_arg_1 = 't1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto'
        cpt_arg_2 = '_default_session_auto.arg'

        if os.path.isdir(os.path.join(self.data_dir, str(sub) + '/')):
            self.surfaces = dict()
            graph_file = os.path.join(self.data_dir, str(sub), cpt_arg_1,
                                      self.side + str(sub) + cpt_arg_2)
            skel_file = os.path.join(
                self.data_dir,
                str(sub),
                self.cpt_skel_1,
                self.side +
                self.cpt_skel_2 +
                str(sub) +
                self.cpt_skel_3)
            graph = aims.read(graph_file)
            self.skel = aims.read(skel_file)

            for v in graph.vertices():
                if 'label' in v:
                    bbmin_surface = v['Tal_boundingbox_min']
                    bbmax_surface = v['Tal_boundingbox_max']
                    bck_map = v['aims_ss']

                    if all([a >= b for (a, b) in zip(bbmin_surface, self.bbmin)]) and all(
                            [a <= b for (a, b) in zip(bbmax_surface, self.bbmax)]):
                        for bucket in bck_map:
                            if bucket.size() > self.ss_size:  # In order to keep only large enough simple surfaces
                                self.surfaces[len(self.surfaces)] = v

            return self.surfaces

    def delete_ss(self, sub):
        """Deletes one simple surface

        Args:
            sub: int giving the subject
        """
        # Suppression of one random simple surface (satisfying both criteria)
        random.seed(42)
        surface = random.randint(0, len(self.surfaces) - 1)
        print(self.surfaces[surface]['label'])

        bck_map = self.surfaces[surface]['aims_ss']
        for voxel in bck_map[0].keys():
            self.skel.setValue(0, voxel[0], voxel[1], voxel[2])

        bck_map_bottom = self.surfaces[surface]['aims_bottom']
        for voxel in bck_map_bottom[0].keys():
            self.skel.setValue(0, voxel[0], voxel[1], voxel[2])

        save_subject = sub
        return save_subject

    def add_ss(self, subjects_list, i):
        """Adds one simple surface from subject i to subject i+1 skeleton

        Args:
            subjects_list: list of subjects
            i: int giving the current subject
        """
        sub_added = subjects_list[i + 1]
        # random.seed(42)
        surface = random.randint(0, len(self.surfaces) - 1)

        if os.path.isdir(self.data_dir + str(sub_added)):
            skel_file = os.path.join(
                self.data_dir,
                str(sub_added),
                self.cpt_skel_1,
                self.side +
                self.cpt_skel_2 +
                str(sub_added) +
                self.cpt_skel_3)
            self.skel = aims.read(skel_file)
            bck_map = self.surfaces[surface]['aims_ss']
            for voxel in bck_map[0].keys():
                if self.skel.value(voxel[0], voxel[1], voxel[2]) != 11:
                    self.skel.setValue(60, voxel[0], voxel[1], voxel[2])

            bck_map_bottom = self.surfaces[surface]['aims_bottom']
            for voxel in bck_map_bottom[0].keys():
                if self.skel.value(voxel[0], voxel[1], voxel[2]) != 11:
                    self.skel.setValue(60, voxel[0], voxel[1], voxel[2])

        save_subject = sub_added
        return save_subject

    def random_skel(self, sub):
        """
        """
        if os.path.isdir(self.data_dir + str(sub)):
            skel_file = os.path.join(
                self.data_dir,
                str(sub),
                self.cpt_skel_1,
                self.side +
                self.cpt_skel_2 +
                str(sub) +
                self.cpt_skel_3)
            self.skel = aims.read(skel_file)

    def save_file(self, sub):
        """Saves the modified skeleton

        Args:
            sub: int giving the subject
        """
        fileout = os.path.join(
            self.saving_dir,
            'output_skeleton_' +
            str(sub) +
            '.nii.gz')
        print('writing altered skeleton to', fileout)
        aims.write(self.skel, fileout)

    def save_lists(self, abnormality_test, givers, subjects_list):
        """Saves lists of modified subjects

        Args:
            abnormality_test: list of modified subjects
            givers: in case of simple surface addition, list of subjects whose
                    one of the simple surface has been added to another subject
            subjects_list: list of subjects
        """
        nor_set = list(set(subjects_list) - set(abnormality_test))

        print('Dataset split, train + val + normal test: ', len(nor_set),
              'abnormal test: ', len(abnormality_test))

        df_train = pd.DataFrame(nor_set)
        df_train.to_csv(os.path.join(self.saving_dir, 'train.csv'))

        df_abnor_test = pd.DataFrame(abnormality_test)
        df_abnor_test.to_csv(
            os.path.join(
                self.saving_dir,
                'abnormality_test.csv'))

        df_givers = pd.DataFrame(givers)
        df_givers.to_csv(os.path.join(self.saving_dir, 'givers.csv'))


def get_sub_list(subjects_list):
    """Returns subjects list for which latered skelerons can be created
    Only right handed HCP subjects
    """
    if subjects_list:
        subjects_list = pd.read_csv(subjects_list)
        subjects_list = list(subjects_list['0'])
    else:
        # Selection of right handed subjects only
        right_handed = pd.read_csv(
            '/neurospin/dico/lguillon/hcp_info/right_handed.csv')
        subjects_list = list(right_handed['Subject'].astype(str))
        # Check whether subjects' files exist
        hcp_sub = os.listdir('/neurospin/hcp/ANALYSIS/3T_morphologist/')
        subjects_list = [sub for sub in subjects_list if sub in hcp_sub]

        random.shuffle(subjects_list)

    return subjects_list


def generate(b_num, side, ss_size, sulci_list, mode='suppress', bench_size=150,
             subjects_list=None, saving_dir=_DEFAULT_SAVING_DIR,
             bbox_dir=_DEFAULT_BBOX_DIR):
    """
    Generates a benchmark

    Args:
        b_num: number of the benchmark to create
        side: hemisphere side (either L for left, or R for right hemisphere)
        sulci_list: list of sulcus names
        mode: string giving the type of benchmark to create ('suppress', 'add'
              or 'mix')
    """
    benchmark = Benchmark(b_num, side, ss_size, sulci_list, saving_dir,
                          bbox_dir=bbox_dir)
    abnormality_test = []
    givers = []
    subjects_list = get_sub_list(subjects_list)

    for i, sub in enumerate(subjects_list):
        print(sub)
        save_sub = sub
        if mode in ['suppress', 'add', 'mix']:
            benchmark.get_simple_surfaces(sub)
            if benchmark.surfaces and len(benchmark.surfaces.keys()) > 0:
                if mode == 'suppress' or (
                        mode == 'mix' and i < bench_size / 2):
                    # Suppression of simple surfaces
                    save_sub = benchmark.delete_ss(sub)
                elif mode == 'add' or (mode == 'mix' and i >= bench_size / 2):
                    # Addition of simple surfaces
                    save_sub = benchmark.add_ss(subjects_list, i)
                    givers.append(sub)
                benchmark.save_file(save_sub)
                # Addition of modified graph to abnormality_test set
                abnormality_test.append(save_sub)
        elif mode == 'random' or mode == 'asymmetry':
            benchmark.random_skel(sub)
            benchmark.save_file(save_sub)
            # Addition of modified graph to abnormality_test set
            abnormality_test.append(save_sub)
        if len(abnormality_test) == bench_size:
            break
    benchmark.save_lists(abnormality_test, givers, subjects_list)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    generate(
        333,
        'R',
        1000,
        sulci_list=[
            'S.T.s.ter.asc.post.',
            'S.T.s.ter.asc.ant.'],
        mode='suppress',
        bench_size=4)
