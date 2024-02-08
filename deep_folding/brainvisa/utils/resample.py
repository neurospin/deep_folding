"""
    Resample a volume that contains discret values
"""
import os
import sys
from contextlib import contextmanager
import logging
from time import time
from typing import Union
import tempfile
import subprocess

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
             values: list = None,
             verbose: bool = False,
             do_skel: bool = False,
             immortals: list = None) -> aims.Volume:
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
            List of unique values ordered by ascendent priority
            without background. If not given,
            priority is set by ascendent values
        do_skel:
            do skeletonization after resampling
        immortals:
            if skeletonization is used, this is the list of voxel values used
            as initial "immortal voxels" in the skeletonization: voxels that
            will not be eroded. Normally border and junctions values, in a
            Morphologist skeleton: [30, 50, 80]. This is the default.

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

    unique_val_in_vol = sorted(np.unique(vol_dt[vol_dt != background]))
    if values is None:
        values = unique_val_in_vol
    else:
        values = [v for v in values if v in unique_val_in_vol]

    # if values is not None, values are given in ascending order
    # Note also that background shall not be given in values

    if aims.version() >= (5, 2):
        reorder = False
        toc = time()
        if values != unique_val_in_vol:
            #print('changing values')
            reorder = True
            # old_vol = vol
            old_resmp = resampled
            resampled = aims.Volume(resampled)
            repl = {v: i+1 for i, v in enumerate(values)}
            repl.update({v: 0 for v in [x for x in np.unique(old_resmp.np)
                                        if x not in values]})
            vol = aims.Volume(vol)  # copy vol
            replacer = getattr(
                aims, 'Replacer_{}'.format(aims.typeCode(old_resmp.np.dtype)))
            replacer.replace(resampled, resampled, repl)
            replacer.replace(vol, vol, repl)
        bck = aims.BucketMap_VOID()
        bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
        cvol_bk = aims.RawConverter_rc_ptr_Volume_S16_BucketMap_VOID(True)
        cvol_bk.convert(vol, bck)
        #for v in bck.keys():
        #    print(v, ':', len(bck[v]), 'voxels')
        t_bck = time() - toc
        toc = time()
        bck2 = aimsalgo.resampleBucket(bck, trm, inv_trm, output_vs)
        t_rs = time() - toc
        toc = time()
        cbk_vol = aims.RawConverter_BucketMap_VOID_rc_ptr_Volume_S16(True)
        cbk_vol.printToVolume(bck2, resampled)
        t_tovol = time() - toc
        # aims.write(resampled, '/tmp/resmp1.nii')  # debug

        if do_skel:
            # skeletonization using Vip command
            # (there are no python bindings for this C library yet)
            # (and VipSkeleton in version 5.1 does not have the -k option)
            if immortals is None:
                immortals = [30, 50, 80]
            immortals_i = [values.index(x) + 1
                          for x in [i for i in immortals if i in values]]
            # skeleton for border lines (junctionos, bottom) will become
            # immortals
            borders = aims.Volume(resampled)
            borders.fill(0)
            for i in immortals_i:
                borders[resampled.np == i] = 1
            tmp = tempfile.mkstemp(prefix='deep_folding_', suffix='.nii')
            os.close(tmp[0])
            tmp2 = tempfile.mkstemp(prefix='deep_folding_sk_', suffix='.nii')
            os.close(tmp2[0])
            tmps = [tmp[1], tmp2[1]]
            try:
                aims.write(borders, tmp[1])
                # aims.write(borders, '/tmp/borders.nii')  # debug
                cmd = ['VipSkeleton', '-i', tmp[1], '-so', tmp2[1], '-fv', 'n',
                       '-sk', 's', '-p', '0', '-c', 'n']
                subprocess.check_call(cmd)
                borders = aims.read(tmp2[1])
                # skeleton of resampled with immortals
                sk_in = aims.Volume(resampled)
                # aims.write(sk_in, '/tmp/borders_sk.nii')  # debug
                sk_in[sk_in.np != 0] = 1
                sk_in[borders.np != 0] = -103  # immportals value
                aims.write(sk_in, tmp[1])
                cmd = ['VipSkeleton', '-i', tmp[1], '-so', tmp2[1], '-fv', 'n',
                       '-sk', 's', '-p', '0', '-c', 'n', '-k']
                subprocess.check_call(cmd)
                del sk_in
                sk_out = aims.read(tmp2[1])
                # aims.write(sk_out, '/tmp/sk.nii')  # debug
                # replace values in skeletonized image
                resampled[sk_out.np == 0] = 0
                del sk_out
            finally:
                for t in tmps:
                    if os.path.exists(t):
                        os.unlink(t)
                    if os.path.exists(t + '.minf'):
                        os.unlink(t + '.minf')

        if reorder:
            print('restoring values')
            reordered = aims.Volume(resampled)
            repl = {i+1: v for i, v in enumerate(values)}
            replacer = getattr(
                aims, 'Replacer_{}'.format(aims.typeCode(old_resmp.np.dtype)))
            old_vals = {v: v for v in np.unique(old_resmp.np)
                        if v not in values}
            replacer.replace(old_resmp, resampled, old_vals)
            replacer.replace(reordered, resampled, repl)

        log.debug("Time: {}s".format(time() - tic))
        log.debug("\t{}s to create the bucket\n\t{}s to resample bucket\n"
                  "\t{}s to assign values".format(t_bck, t_rs, time() - toc))

    else:
        if do_skel:
            print('WARNING: skeletonization is only available with aims/vip '
                  '>= 5.2. You are using {} so this step will not be '
                  'done.'.format(aims.__version__))
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
                v, np.where(vol_dt == v)[0].shape[0], time() - tic))
            log.debug("\t{}s to create the bucket\n\t{}s to resample bucket\n"
                      "\t{}s to assign values".format(t_bck, t_rs, time() - toc))
            tic = time()

    return resampled
