# -*- coding: utf-8 -*-
# /usr/bin/env python3
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

"""
The aim of this script is to create pytorch dataloaders from MRIs saved as
numpy arrays in a .pickle.
"""
import os
import time
from datetime import date
import argparse
import pandas as pd
import itertools

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from .datasets import SkeletonDataset, AugDatasetTransformer
from ..utils import save_results


def create_hcp_benchmark(side, benchmark, directory, batch_size, handedness=1):
    """
    Creates datasets from HCP data and depending on dataset split of benchmark
    generation (cf anatomist_tools.benchmark_generation module)
    /!\ ONLY DIFFERENCE FROM create_hcp_sets function is only that it creates
    sets from benchmark split.
    IN: side: str, 'right' or 'left'
        handedness: int, 1 if right handed, 2 if left handed
        directory: str, folder in which save the results
        batch_size: int, size of training batches
    OUT: root_dir,
         dataset_train_loader,
         dataset_val_loader,
         dataset_test_loader
    """
    date_exp = date.today().strftime("%d%m%y")

    #train_list = pd.read_csv('/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark' + str(benchmark) + '/0_Rside/train.csv')
    train_list = pd.read_csv('/neurospin/dico/lguillon/benchmark/sc/benchmark' + str(benchmark) + '/train.csv')
    train_list = train_list.rename(columns={"0":"Subject"})

    loss_type = 'CrossEnt'
    root_dir = directory + side + '_hemi_skeleton_' + date_exp + '_' +loss_type + '_' + str(handedness) + '_3classes/'
    print(root_dir)
    save_results.create_folder(root_dir)

    #data_dir = '/neurospin/dico/lguillon/skeleton/sts_crop/'
    data_dir = '/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm/'
    input_data = 'Rskeleton'
    tmp = pd.read_pickle(data_dir + input_data +'.pkl')
    tmp = tmp.T
    tmp = tmp.rename(columns={0:'ID'})
    train = pd.merge(tmp, train_list.Subject.astype(str), left_on = tmp.index, right_on='Subject')
    train = train.reset_index(drop=True)
    filenames = list(train.Subject)

    hcp_dataset_train = SkeletonDataset(dataframe=train, filenames=filenames)
    print(len(hcp_dataset_train))
    # Split training set into train, val and test
    partition = [0.7, 0.2, 0.1]

    random_seed = 42
    torch.manual_seed(random_seed)

    print([round(i*(len(hcp_dataset_train))) for i in partition])
    train_set, val_set, test_set = torch.utils.data.random_split(hcp_dataset_train,
                         [round(i*(len(hcp_dataset_train))) for i in partition])

    # Data Augmentation application
    train_set = AugDatasetTransformer(train_set)

    return root_dir, train_set, val_set, test_set


def create_benchmark_test(benchmark, side, handedness=1):
    """
    Creates test datasets from benchmark of altered skeletons
    (cf anatomist_tools.benchmark_generation module)
    IN: benchmark: int, number of benchmark
        handedness: int, 1 if right handed, 2 if left handed
    OUT: dataset_test_abnor_loader
    """
    data_dir = '/neurospin/dico/lguillon/benchmark/sc/benchmark' + str(benchmark) + '/'

    input_data = 'abnormal_skeleton_' + side
    print(input_data)
    tmp = pd.read_pickle(data_dir + input_data +'.pkl')
    filenames = list(tmp.columns)
    tmp = tmp.T

    benchmark_dataset = SkeletonDataset(dataframe=tmp, filenames=filenames)

    return benchmark_dataset


def create_hcp_sets(skeleton, side, directory, batch_size, handedness=0):
    """
    Creates datasets from HCP data
    IN: skeleton: boolean, True if input is skeleton, False otherwise,
        side: str, 'right' or 'left'
        handedness: int, 0 if mixed ind, 1 if right handed, 2 if left handed
        directory: str, folder in which save the results
        batch_size: int, size of training batches
        weights: list, list of weights to apply to skeleton values
    OUT: root_dir: created directory where results will be stored
         dataset_train_loader, dataset_val_loader, dataset_test_loader: loaders
         that will be used for training and testing
    """
    print(torch.cuda.current_device())
    date_exp = date.today().strftime("%d%m%y")
    if skeleton == True:
        skel = 'skeleton'
        loss_type = 'CrossEnt'
        root_dir = directory + side + '_hemi_' + skel + '_' + date_exp + '_' +loss_type + '_' + str(handedness) + '_2classes/'
    else:
        skel = 'norm_spm'
        loss_type = 'L2'
        root_dir = directory + side + '_hemi_' + skel + '_' + date_exp + '_' +loss_type + '_' + str(handedness) +'/'

    #print("Parameters : skeleton: {}, side: {}, weights: {}, loss_type: {}".format(skeleton, side, weights, loss_type))
    print(root_dir)
    save_results.create_folder(root_dir)

    if skeleton:
        data_dir = '/neurospin/dico/lguillon/skeleton/sts_crop/'
        #data_dir = '/home_local/lg261972/data/'
        if handedness == 0:
            input_data = 'sts_crop_skeleton_' + side
            tmp = pd.read_pickle(data_dir + input_data +'.pkl')
            filenames = list(tmp.columns)
            tmp = torch.from_numpy(np.array([tmp.loc[0].values[k] for k in range(len(tmp))]))
        else:
            if handedness == 1:
                input_data = side + '_hemi_rightH_sts_crop_skeleton'
            else:
                input_data = side + '_hemi_leftH_sts_crop_skeleton'
            print(input_data)
            tmp = pd.read_pickle(data_dir + input_data +'.pkl')
            filenames = tmp.Subject.values
            print(len(filenames))
            tmp = torch.from_numpy(np.array([tmp.loc[k].values[0] for k in range(len(tmp))]))

    else:
        data_dir = '/neurospin/dico/lguillon/hcp_cs_crop/sts_crop/'+ side + '_hemi/'
        data_dir = '/home_local/lg261972/data/'
        if handedness == 0:
            input_data = 'sts_crop_' + side
            tmp = pd.read_pickle(data_dir + input_data +'.pkl')
            filenames = list(tmp.columns)
            tmp = torch.from_numpy(np.array([tmp.loc[0].values[k] for k in range(len(tmp))]))
        else:
            if handedness == 1:
                input_data = side + '_hemi_rightH_sts_crop'
            else:
                input_data = side + '_hemi_leftH_sts_crop'
            print(input_data)
            tmp = pd.read_pickle(data_dir + input_data +'.pkl')
            filenames = tmp.Subject.values
            print(len(filenames))
            tmp = torch.from_numpy(np.array([tmp.loc[k].values[0] for k in range(len(tmp))]))

    tmp = tmp.to('cuda')

    hcp_dataset = TensorDataset(filenames=filenames, data_tensor=tmp,
                                skeleton=skeleton, vae=False)
    # Split training set into train, val and test
    partition = [0.7, 0.2, 0.1]
    print([round(i*(len(hcp_dataset))) for i in partition])
    train_set, val_set, test_set = torch.utils.data.random_split(hcp_dataset,
                            [round(i*(len(hcp_dataset))) for i in partition])

    #train_set = AugDatasetTransformer(train_set)
    #val_set = AugDatasetTransformer(val_set)
    #test_set  = AugDatasetTransformer(test_set)

    dataset_train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
    dataset_val_loader = torch.utils.data.DataLoader(val_set, shuffle=True,
                                                          num_workers=0)
    dataset_test_loader = torch.utils.data.DataLoader(test_set, shuffle=True,
                                                          num_workers=0)

    print("Dataset generated \n Size of training dataset :", len(dataset_train_loader))

    return root_dir, dataset_train_loader, dataset_val_loader, dataset_test_loader


def create_left_handed_set(skeleton, side):
    """
    Creates dataset of left handed HCP individuals
    IN: skeleton: boolean, True if input is skeleton, False otherwise,
        side: str, corresponds to hemisphere's side 'right' or 'left'
    OUT:
    """

    data_dir = '/home_local/lg261972/data/'
    hand = 'left'

    if skeleton:
        #hcp_leftH = pd.read_pickle('/home_local/lg261972/data/data_handedness/{}_hemi/controls_qc_{}_{}H.pkl'.format(side, side, hand))
        input_data = side + '_hemi_leftH_sts_crop_skeleton'
        print(input_data)
        tmp = pd.read_pickle(data_dir + input_data +'.pkl')
        filenames = tmp.Subject.values
        print(len(filenames))
        tmp = torch.from_numpy(np.array([tmp.loc[k].values[0] for k in range(len(tmp))]))
    else:
        hcp_leftH = pd.read_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/{}_hemi/controls_qc_{}_{}H.pkl'.format(side, side, hand))

    tmp = tmp.to('cuda')

    leftH_hcp_dataset = TensorDataset(filenames=filenames, data_tensor=tmp,
                                    skeleton=skeleton, vae=False)

    leftH_loader = torch.utils.data.DataLoader(leftH_hcp_dataset, batch_size=1,
                                                    shuffle=True, num_workers=0)

    return leftH_loader


def create_loader_from_csv(subject_list, side):
    """
    Creates a dataloader from a list of subjects
    IN: subject_list: list of subjects to put in loader, csv file
        side: 'left' or 'right' hemisphere, str
    OUT: dataloader
    """

    subject_list = pd.read_csv(subject_list)

    data_dir = '/neurospin/dico/lguillon/skeleton/sts_crop/'
    input_data = side + '_hemi_rightH_sts_crop_skeleton'
    tmp = pd.read_pickle(data_dir + input_data +'.pkl')

    subject_dataset = pd.merge(tmp, subject_list.subjects.astype(str), left_on='Subject',
                     right_on='subjects')
    subject_dataset = subject_dataset.reset_index(drop=True)
    filenames = list(subject_dataset.Subject)

    subject_dataset = SkeletonDataset(dataframe=subject_dataset, filenames=filenames)

    subject_loader = torch.utils.data.DataLoader(subject_dataset, batch_size=1,
                                                    shuffle=True, num_workers=0)

    return subject_loader
