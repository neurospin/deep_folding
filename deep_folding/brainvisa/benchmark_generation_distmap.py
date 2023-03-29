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
from deep_folding.brainvisa.utils.mask import compute_simple_mask
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.brainvisa.utils.skeleton import generate_skeleton_from_graph
from deep_folding.brainvisa.utils.foldlabel import \
    generate_foldlabel_thin_junction
import dico_toolbox as dtx


_DEFAULT_DATA_DIR = '/neurospin/dico/data/bv_databases/human/hcp/hcp'
_DEFAULT_MASK_DIR = '/neurospin/dico/data/deep_folding/current/mask/1mm/'
_DEFAULT_SAVING_DIR = '/neurospin/dico/lguillon/distmap/benchmark/test/'
_DEFAULT_BBOX_DIR = '/neurospin/dico/data/deep_folding/current/bbox/'


class Benchmark():
    """Generates benchmark of altered skeletons
    """

    def __init__(self, b_num, side, ss_size, sulci_list, saving_dir,
                 data_dir=_DEFAULT_DATA_DIR, bbox_dir=_DEFAULT_BBOX_DIR,
                 mask_dir=_DEFAULT_MASK_DIR, mask=True, inpainting=False):
        """Inits with list of directories, bounding box and sulci

        Args:
            b_num: number of benchmark
            side: hemisphere side (L for left, or R for right hemisphere)
            ss_size: minimum size of simple surface to consider
            sulci_list: list of sulcus names
            data_dir: string naming full path source directories containing
                      MRI images
            saving_dir: name of directory where altered skeletons will be saved
            mask_dir: name of directory where masks are stored
        """
        self.inpainting = inpainting
        self.b_num = b_num
        self.side = side
        self.ss_size = ss_size
        self.sulci_list = complete_sulci_name(sulci_list, self.side)
        self.src_dir = data_dir
        self.saving_dir = saving_dir
        self.saving_dir_skel = saving_dir + 'skeletons/raw/'
        self.saving_dir_foldlabel = saving_dir + 'foldlabels/raw/'
        self.path_to_graph = "t1mri/BL/default_analysis/folds/3.1/deepcnn_auto"
        self.abnormality_test = []
        self.voxel_size_out = (1, 1, 1, 1)
        if mask:
            # Get mask and corresponding bounding_box
            self.mask, self.bbmin, self.bbmax = compute_simple_mask(
                self.sulci_list, side, mask_dir=mask_dir)
        else:
            self.bbmin, self.bbmax = compute_max_box(
                self.sulci_list, side, talairach_box=True, src_dir=bbox_dir)

        print(self.mask, self.bbmin, self.bbmax)

    def get_simple_surfaces(self, sub):
        """Selects simple surfaces of one subject that satisfy following
           conditions: size in [given ss_size min, given ss_size max] and with
           the right number of voxels included in the mask

        Args:
            sub: int giving the subject
        """

        self.graph_file = f"{self.src_dir}/{sub}/" +\
            f"{self.path_to_graph}/{self.side}{sub}_deepcnn_auto.arg"

        if os.path.isdir(os.path.join(self.src_dir, str(sub) + '/')) \
                and os.path.isfile(self.graph_file):
            self.surfaces = dict()

            self.graph = aims.read(self.graph_file)

            g_to_icbm_template = aims.GraphManip.getICBM2009cTemplateTransform(
                self.graph)
            voxel_size_in = self.graph['voxel_size'][:3]

            for v in self.graph.vertices():
                if 'label' in v:
                    bck_map = v['aims_ss']
                    # Creation of a volume in ICBM space where to write voxels
                    # of the simple surface
                    hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
                    resampling_ratio = np.array(
                        hdr['voxel_size']) / self.voxel_size_out
                    orig_dim = hdr['volume_dimension']
                    new_dim = list((resampling_ratio * orig_dim).astype(int))

                    vol = aims.Volume(new_dim, dtype='S16')
                    vol.copyHeaderFrom(hdr)
                    vol.header()['voxel_size'] = self.voxel_size_out
                    arr = np.asarray(vol)
                    # Transformation of SS voxels to ICBM space with
                    # voxel_size_out
                    voxels_icbm = np.asarray(
                        [g_to_icbm_template.transform(
                            np.array(voxel) * voxel_size_in
                            )
                         for voxel in bck_map[0].keys()])
                    voxels = np.round(np.array(voxels_icbm) /
                                      self.voxel_size_out[:3]).astype(int)
                    # Writing of the voxels in the created volume
                    for i, j, k in voxels:
                        arr[i, j, k, 0] = 1
                    # Suppression of all voxels out of the mask
                    arr[np.array(self.mask) < 1] = 0
                    # Selection of the ss if a mininum of voxels remains
                    if np.count_nonzero(
                            arr > 0) > self.ss_size and np.count_nonzero(
                            arr > 0) < 500:
                        print(v['label'], bck_map[0].size())
                        print(np.count_nonzero(arr == 1))
                        self.surfaces[len(self.surfaces)] = v

            return self.surfaces

    def delete_ss(self, sub):
        """Deletes one simple surface

        Args:
            sub: int giving the subject
        """
        # Suppression of one random simple surface (satisfying both criteria)
        random.seed(56)  # benchmark 1 : random seed = 42
        surface = random.randint(0, len(self.surfaces) - 1)

        bck_map = self.surfaces[surface]['aims_ss']
        for voxel in bck_map[0].keys():
            self.skel.setValue(0, voxel[0], voxel[1], voxel[2])
            if self.inpainting:
                self.foldlabel.setValue(0, voxel[0], voxel[1], voxel[2])

        bck_map_bottom = self.surfaces[surface]['aims_bottom']
        for voxel in bck_map_bottom[0].keys():
            self.skel.setValue(0, voxel[0], voxel[1], voxel[2])
            if self.inpainting:
                self.foldlabel.setValue(0, voxel[0], voxel[1], voxel[2])

        for k in range(len(self.surfaces[surface].edges())):
            if 'aims_junction' in self.surfaces[surface].edges()[k]:
                bck_map_junction = self.surfaces[surface].edges()[
                    k]['aims_junction']
                for voxel in bck_map_junction[0].keys():
                    self.skel.setValue(0, voxel[0], voxel[1], voxel[2])
                    if self.inpainting:
                        self.foldlabel.setValue(
                            0, voxel[0], voxel[1], voxel[2])

        save_subject = sub
        return save_subject

    def generate_skeleton(self, sub):
        """Generates a skeleton from a graph

        Args:
            sub: int giving the subject
        """
        self.skel = generate_skeleton_from_graph(self.graph)

    def generate_foldlabel(self, sub):
        """Deletes one simple surface
        Args:
            sub: int giving the subject
        """
        self.foldlabel = generate_foldlabel_thin_junction(self.graph)

    def save_file(self, sub):
        """Saves the modified skeleton

        Args:
            sub: int giving the subject
        """
        fileout = os.path.join(
            self.saving_dir,
            'modified_skeleton_' +
            str(sub) +
            '.nii.gz')
        print('writing altered skeleton to', fileout)
        aims.write(self.skel, fileout)
        if self.inpainting:
            fileout = os.path.join(
                self.saving_dir,
                'modified_foldlabel_' +
                str(sub) +
                '.nii.gz')
            print('writing altered foldlabel to', fileout)
            aims.write(self.foldlabel, fileout)

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
        subjects_list = list(subjects_list['subjects'])
    else:
        # Selection of right handed subjects only
        right_handed = pd.read_csv(
            '/neurospin/dico/lguillon/hcp_info/right_handed.csv')
        subjects_list = list(right_handed['subjects'].astype(str))
        # Check whether subjects' files exist
        hcp_sub = os.listdir(_DEFAULT_DATA_DIR)
        subjects_list = [sub for sub in subjects_list if sub in hcp_sub]

        random.shuffle(subjects_list)

    return subjects_list


def generate(b_num, side, ss_size, sulci_list, bench_size=150,
             subjects_list=None, saving_dir=_DEFAULT_SAVING_DIR,
             bbox_dir=_DEFAULT_BBOX_DIR, inpainting=False):
    """
    Generates a benchmark

    Args:
        b_num: number of the benchmark to create
        side: hemisphere side (either L for left, or R for right hemisphere)
        sulci_list: list of sulcus names
    """
    benchmark = Benchmark(b_num, side, ss_size, sulci_list, saving_dir,
                          bbox_dir=bbox_dir, inpainting=inpainting)
    abnormality_test = []
    givers = []
    subjects_list = get_sub_list(subjects_list)

    for i, sub in enumerate(subjects_list):
        print(sub)
        save_sub = sub
        benchmark.get_simple_surfaces(sub)
        if benchmark.surfaces and len(benchmark.surfaces.keys()) > 0:
            benchmark.generate_skeleton(sub)
            benchmark.generate_foldlabel(sub)
            # Suppression of simple surfaces
            save_sub = benchmark.delete_ss(sub)

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
        200,
        sulci_list=['S.C.'],
        subjects_list='/neurospin/dico/lguillon/distmap/data/test_list.csv',
        bench_size=3,
        inpainting=False)
