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
Tools in order to create pytorch dataloaders
"""

import pandas as pd
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate

from .pynet_transforms import *

class TensorDataset():
    """Custom dataset that includes image file paths.
    Applies different transformations to data depending on the type of input.
    IN: data_tensor: tensor containing MRIs as numpy arrays
        filenames: list of subjects' IDs
        skeleton: boolean, whether input is skeleton images or not
    OUT: tensor of [batch, sample, subject ID]
    """
    def __init__(self, data_tensor, filenames, skeleton):
        self.skeleton = skeleton
        self.data_tensor = data_tensor
        self.transform = True
        self.nb_train = len(filenames)
        print(self.nb_train)
        self.filenames = filenames
        self.vae = vae

    def __len__(self):
        return(self.nb_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_tensor[idx]
        file = self.filenames[idx]

        if self.skeleton:
            fill_value = 1
            sample = NormalizeSkeleton(sample)()
            self.transform = transforms.Compose([DownsampleTensor(scale=2),
                        PaddingTensor([1, 40, 40, 40], fill_value=fill_value)
            ])

            sample = self.transform(sample)

        else:
            self.transform = PaddingTensor([1, 80, 80, 80])
            sample = self.transform(sample)
            self.transform = DownsampleTensor(scale=2)
            sample = self.transform(sample)
            if self.vae:
                self.transform = DownsampleTensor(scale=2)
                sample = self.transform(sample)
            sample = NormalizeHisto(sample)()

        tuple_with_path = (sample, file)
        return tuple_with_path


class SkeletonDataset():
    """Custom dataset for skeleton images that includes image file paths.
    dataframe: dataframe containing training and testing arrays
    filenames: optional, list of corresponding filenames
    Works on CPUs
    """
    def __init__(self, dataframe, filenames=None):
        self.df = dataframe
        if filenames:
            self.filenames = filenames
            #self.df = self.df.T
        else:
            self.filenames = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.filenames:
            filename = self.filenames[idx]
            sample = self.df.iloc[idx][0]
        else:
            filename = self.df.iloc[idx]['ID']
            sample = self.df.iloc[idx][0]

        fill_value = 1
        sample = NormalizeSkeleton(sample)()
        self.transform = transforms.Compose([Downsample(scale=2),
                         Padding([1, 40, 40, 40], fill_value=fill_value)
                         ])
        sample = self.transform(sample)
        tuple_with_path = (sample, filename)
        return tuple_with_path


class AugDatasetTransformer(torch.utils.data.Dataset):
    """
    Custom dataset that applies data augmentation on a dataset processed
    through TensorDataset or Skeleton Dataset classes.
    Transformations are performed on CPU.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        img, filename = self.base_dataset[index]
        if np.random.rand() > 0.6:
            self.angle = np.random.randint(-90, 90)
            img = np.expand_dims(rotate(img[0], angle=self.angle, reshape=False, cval=1, order=1), axis=0)
        return img, filename

    def __len__(self):
        return len(self.base_dataset)
