"""
    Resample a volume that contains discret values
"""
import os
import sys
from contextlib import contextmanager
import logging
from time import time
from typing import Union

import numpy as np
from soma import aims
from soma import aimsalgo

from deep_folding.config.logs import set_file_logger
# Defines logger
log = set_file_logger(__file__)


def resample(input_image: Union[str, aims.Volume],
             transformation: Union[str, aims.AffineTransformation3d],
             output_vs: tuple = None,
             background: int = 0,
             values: np.array = None,
             verbose: bool = False) -> aims.Volume:
    """
        Transforms and resamples a volume that has discret values

        Parameters
        ----------
        input_image: path to nifti file or aims volume
            Path to the input volume (.nii or .nii.gz file)
        transformation: path to file
            Linear transformation file (.trm file)
        output_vs: tuple
            Output voxel size (default: None, no resampling)
        background: int
            Background value (default: 0)
        values: []
            Array of unique values ordered by ascendent priority
            without background. If not given,
            priority is set by ascendent values

        Return
        ------
        resampled:
            Transformed and resampled volume
    """

    global log

    # Handling of verbosity
    if verbose:
        log.setLevel(level=logging.DEBUG)

    tic = time()

    # Reads input image (either path to file or aims volume)
    if isinstance(input_image, str):
        vol = aims.read(input_image)
    else:
        vol = input_image
    vol_dt = vol.__array__()

    # Reads transformation if present (either path to file or aims Volume)
    if transformation:
        if isinstance(transformation, str):
            trm = aims.read(transformation)
        else:
            trm = transformation
    else:
        trm = aims.AffineTransformation3d(np.eye(4))
    inv_trm = trm.inverse()

    ##################################
    # Definition of voxel size and dim
    ##################################

    if output_vs:
        output_vs = np.array(output_vs)
        hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
        # New volume dimensions
        resampling_ratio = np.array(hdr['voxel_size'][:3]) / output_vs
        orig_dim = hdr['volume_dimension'][:3]
        new_dim = list((resampling_ratio * orig_dim).astype(int))
    else:
        output_vs = vol.header()['voxel_size'][:3]
        new_dim = vol.header()['volume_dimension'][:3]

    log.debug("Time before resampling: {}s".format(time() - tic))
    tic = time()

    # Transform the background
    # Using the inverse is more straightforward and supports non-linear
    # transforms
    resampled = aims.Volume(new_dim, dtype=vol_dt.dtype)
    resampled.copyHeaderFrom(hdr)
    resampled.header()['voxel_size'] = output_vs
    # 0 order (nearest neightbours) resampling
    resampler = aimsalgo.ResamplerFactory(vol).getResampler(0)
    resampler.setDefaultValue(background)
    resampler.setRef(vol)
    resampler.resample_inv(vol, inv_trm, 0, resampled, True)
    resampled_dt = np.asarray(resampled)

    log.debug("Background resampling: {}s".format(time() - tic))
    tic = time()

    if values is None:
        values = sorted(np.unique(vol_dt[vol_dt != background]))

    # if values is not None, values are given in ascending order
    # Note also that background shall not be given in values

    # Create one bucket by value (except background)
    # FIXME: Create several buckets because I didn't understood how to add
    # several bucket to a BucketMap
    for i, v in enumerate(values):
        toc = time()
        bck = aims.BucketMap_VOID()
        bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
        bk0 = bck[0]
        for p in np.vstack(np.where(vol_dt == v)[:3]).T:
            bk0[list(p)] = v
        t_bck = time() - toc
        toc = time()
        bck2 = aimsalgo.resampleBucket(bck, trm, inv_trm, output_vs)
        t_rs = time() - toc
        toc = time()
        # FIXME: Could not assign the correct value with the converter.
        # TODO: try to reduce time consumation of this part!
        # Using the converter, the new_dim must incremented
        # conv = aims.Converter(intype=bck2, outtype=aims.AimsData(vol))
        # conv.convert(bck2, resampled)
        # Use a for loop instead:
        for p in bck2[0].keys():
            c = p.list()
            if c[0] < new_dim[0] and c[1] < new_dim[1] and c[2] < new_dim[2]:
                resampled_dt[c[0], c[1], c[2]] = values[i]

        log.debug("Time for value {} ({} voxels): {}s".format(
            v, np.sum(np.where(vol_dt == v)), time() - tic))
        log.debug("\t{}s to create the bucket\n\t{}s to resample bucket\n"
                  "\t{}s to assign values".format(t_bck, t_rs, time() - toc))
        tic = time()

    return resampled
