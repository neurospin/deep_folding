# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Module that defines common transformations that can be applied when the dataset
is loaded.
"""

# Imports
import collections
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func


class PaddingTensor(object):
    """A class to pad a tensor"""
    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.
        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = shape
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, tensor):
        """ Fill a tensor to fit the desired shape.
        Parameters
        ----------
        tensor: torch.tensor
            an input tensor.
        Returns
        -------
        fill_tensor: torch.tensor
            the fill_value padded tensor.
        """
        if len(tensor.shape) - len(self.shape) == 1:
            data = []
            for _tensor, _fill_value in zip(tensor, self.fill_value):
                data.append(self._apply_padding(_tensor, _fill_value))
            return np.asarray(data)
        elif len(tensor.shape) - len(self.shape) == 0:
            return self._apply_padding(tensor, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, tensor, fill_value):
        """ See Padding.__call__().
        """
        orig_shape = tensor.shape
        padding = ()
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0 and shape_i != 0:
                padding += (half_shape_i, half_shape_i)
            elif shape_i % 2 ==1:
                padding += (half_shape_i, half_shape_i + 1)
        for cnt in range(len(tensor.shape) - len(padding)):
            padding += (0, 0)
        padding = padding[::-1]
        fill_tensor = func.pad(tensor, padding, "constant", fill_value)
        return fill_tensor


class DownsampleTensor(object):
    """ A class to downsample a tensor.
    """
    def __init__(self, scale, with_channels=True):
        """ Initialize the instance.
        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        with_channels: bool, default True
            if set expect the array to contain the channels in first dimension.
        """
        self.scale = scale
        self.with_channels = with_channels

    def __call__(self, tensor):
        """ Downsample a tensor to fit the desired shape.
        Parameters
        ----------
        tensor: torch.tensor
            an input tensor
        Returns
        -------
        down_tensor: torch.tensor
            the downsampled tensor.
        """
        if self.with_channels:
            data = []
            for _tensor in tensor:
                data.append(self._apply_downsample(_tensor))
            return data[0].unsqueeze(0)
        else:
            return self._apply_downsample(tensor)

    def _apply_downsample(self, tensor):
        """ See Downsample.__call__().
        """
        slices = []
        for cnt, orig_i in enumerate(tensor.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_tensor = tensor[tuple(slices)]

        return down_tensor


class Padding(object):
    """ A class to pad an image.
    """
    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.
        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = shape
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.
        Parameters
        ----------
        arr: np.array
            an input array.
        Returns
        -------
        fill_arr: np.array
            the zero padded array.
        """
        if len(arr.shape) - len(self.shape) == 1:
            data = []
            for _arr, _fill_value in zip(arr, self.fill_value):
                data.append(self._apply_padding(_arr, _fill_value))
            return np.asarray(data)
        elif len(arr.shape) - len(self.shape) == 0:
            return self._apply_padding(arr, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr, fill_value):
        """ See Padding.__call__().
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))
        #print(padding)
        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)
        return fill_arr


class Downsample(object):
    """ A class to downsample an array.
    """
    def __init__(self, scale, with_channels=True):
        """ Initialize the instance.
        Parameters
        ----------
        scale: int
            the downsampling scale factor in all directions.
        with_channels: bool, default True
            if set expect the array to contain the channels in first dimension.
        """
        self.scale = scale
        self.with_channels = with_channels

    def __call__(self, arr):
        """ Downsample an array to fit the desired shape.
        Parameters
        ----------
        arr: np.array
            an input array
        Returns
        -------
        down_arr: np.array
            the downsampled array.
        """
        if self.with_channels:
            data = []
            for _arr in arr:
                data.append(self._apply_downsample(_arr))
            return np.asarray(data)
        else:
            return self._apply_downsample(arr)

    def _apply_downsample(self, arr):
        """ See Downsample.__call__().
        """
        slices = []
        for cnt, orig_i in enumerate(arr.shape):
            if cnt == 3:
                break
            slices.append(slice(0, orig_i, self.scale))
        down_arr = arr[tuple(slices)]

        return down_arr


class NormalizeHisto(object):
    """
    Class to normalize voxels' intensity values to interval
    """

    def __init__(self, arr, *args):
        """ Initialize the instance"""
        self.arr = arr

    def __call__(self):
        """
        Normalize an array
        Parameters
        ----------
        arr: np.array, an input array
        Returns
        -------
        norm_arr: np.array
            normalized array
        """
        # Before normalization
        # flat = self.arr.flatten()
        # plt.hist(flat, bins=50)
        # plt.show()
        self.arr = self.arr.float()
        ave = self.arr.mean()
        std = self.arr.std()
        self.arr = (self.arr - ave)/std
        # After normalization
        # flat2 = self.arr.flatten()
        # plt.hist(flat2, bins=100)
        # plt.show()

        return self.arr

class NormalizeSkeleton(object):
    """
    Class to normalize skeleton objects,
    black voxels: 0
    grey and white voxels: 1
    """
    def __init__(self, arr, *args):
        """ Initialize the instance"""
        self.arr = arr
        # self.maximum = self.arr.max()

    def __call__(self):
        # self.arr = self.arr/self.maximum
        #self.arr[self.arr == 11] = 1
        #self.arr[self.arr >= 10] = 2
        #print(torch.unique(self.arr))
        #print("Here.... Transformation ! ")
        """self.arr[self.arr < 11] = 0
        self.arr[self.arr > 11] = 2
        self.arr[self.arr == 11] = 1"""
        # With 3 classes
        #print(type(np.unique(self.arr.cpu().numpy())))
        """if len(np.unique(self.arr.cpu().numpy()))>3:
            self.arr[self.arr < 11] = 0 # inside the brain
            self.arr[self.arr > 11] = 2 # sulci
            self.arr[self.arr == 11] = 1 # out of the brain"""
        # With only 2 classes:
        self.arr[self.arr == 0] = 0 # inside the brain
        self.arr[self.arr > 0] = 1 # sulci + out of the brain
        #print(torch.unique(self.arr))

        return self.arr
