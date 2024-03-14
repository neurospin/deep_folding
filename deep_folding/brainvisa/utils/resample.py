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
             immortals: list = None,
             redo_classif: bool = True) -> aims.Volume:
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
            without background. If not given, priority is set by ascending
            values. For Morphologist or deep_folding skeletons (see
            :func:`~deep_folding.brainvisa.utils.skeleton.generate_skeleton_thin_junction`) a good values list is:
            [100, 60, 10, 20, 40, 50, 70, 80, 110, 120, 30, 35].
        do_skel:
            do skeletonization after resampling
        immortals:
            if skeletonization is used, this is the list of voxel values used
            as initial "immortal voxels" in the skeletonization: voxels that
            will not be eroded. Normally border and junctions values, in a
            Morphologist or deep_folding skeleton: [30, 50, 80, 35, 110, 120].
            This is the default.
        redo_classif:
            if True, after re-skeletonization, re-perform voxel topological
            classification. No effect if do_skel is False.

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
        vol = aims.Volume(input_image)
    vol_dt = vol.np

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

    hdr = aims.StandardReferentials.icbm2009cTemplateHeader()
    if output_vs:
        output_vs = np.array(output_vs)
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
    resampled.fill(0)

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

        if immortals is None:
            immortals = [30, 50, 80, 35, 110, 120]
        immortals = [v for v in values if v in immortals]
        values = [v for v in values if v not in immortals] \
            + [v for v in immortals]

        # make sure the background is 0 before transformation into bucket
        # FIXME: what if 0 is actually a non-background label ?
        vol[vol.np == background] = 0

        if do_skel:
            cc_vol = aims.Volume(vol)
            cc_vol[~np.isin(cc_vol.np, immortals)] = 0

            aims.AimsConnectedComponent(
                cc_vol, aims.Connectivity.CONNECTIVITY_26_XYZ, 0,
                False, 0, 0, 0, False)
            bck = aims.BucketMap_VOID()
            bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
            cvol_bk = aims.RawConverter_rc_ptr_Volume_S16_BucketMap_VOID(True)
            cvol_bk.convert(cc_vol, bck)

            cc_per_v = {}
            for cc, pts in bck.items():
                p = tuple(pts.keys()[0]) + (0, )
                cc_per_v.setdefault(vol[p], []).append(cc)

            repl = {}
            repl_cc = {}
            i = 1
            for v in values:
                if v not in immortals:
                    repl[v] = i
                    i += 1
                else:
                    for vv in cc_per_v[v]:
                        repl_cc[vv] = i
                        i += 1
        else:
            repl = {v: i+1 for i, v in enumerate(values)}
            repl_cc = {}
            cc_per_v = {}
            cc_vol = vol

        replacer = getattr(
            aims, 'Replacer_{}'.format(aims.typeCode(cc_vol.np.dtype)))
        if repl_cc:
            replacer.replace(cc_vol, cc_vol, repl_cc)
        repl.update({v: 0 for v in [x for x in unique_val_in_vol
                                    if x not in values]})
        replacer.replace(vol, cc_vol, repl)

        reorder = True

        toc = time()

        bck = aims.BucketMap_VOID()
        bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
        cvol_bk = aims.RawConverter_rc_ptr_Volume_S16_BucketMap_VOID(True)
        cvol_bk.convert(cc_vol, bck)
        t_bck = time() - toc
        toc = time()
        # resampleBucket() will process values in ascending order, thus the
        # values order is very important.
        bck2 = aimsalgo.resampleBucket(bck, trm, inv_trm, output_vs)
        t_rs = time() - toc
        toc = time()
        cbk_vol = aims.RawConverter_BucketMap_VOID_rc_ptr_Volume_S16(True)
        cbk_vol.printToVolume(bck2, resampled)
        t_tovol = time() - toc

        if do_skel:
            # skeletonization using Vip command
            # (there are no python bindings for this C library yet)
            # (and VipSkeleton in version 5.1 does not have the -k option)
            immortals_i = range(repl_cc[cc_per_v[immortals[0]][0]],
                                repl_cc[cc_per_v[immortals[-1]][-1]] + 1)
            unique_rsp = np.unique(resampled)
            immortals_i = [v for v in immortals_i if v in unique_rsp]

            # skeleton for border lines (junctions, bottom) will become
            # immortals
            borders = aims.Volume(resampled)
            borders[~np.isin(borders.np, immortals_i)] = 0

            # we must skeletonize the border lines first. For this, we also
            # need to set the lines extremities as immortals. So we must
            # get connected components, set a distance map seed in each cc,
            # make a distance map, get the max for each cc, then do another
            # distance map to get the second extremity of each cc.
            aims.AimsConnectedComponent(
                borders, aims.Connectivity.CONNECTIVITY_26_XYZ, 0, False, 0, 0,
                0, False)
            ccn = np.max(borders.np) + 1
            bck = aims.BucketMap_VOID()
            cvol_bk = aims.RawConverter_rc_ptr_Volume_S16_BucketMap_VOID(True)
            cvol_bk.convert(borders, bck)
            pts = []
            for cc, b in bck.items():
                pts.append(b.keys()[0])
            borders_cc = aims.Volume(borders)
            borders[borders.np != 0] = 1
            borders[tuple(np.array(pts).T)] = np.expand_dims(
                np.arange(2, ccn + 1), 1)
            aimsalgo.AimsDistanceFrontPropagation(borders, 1, 0, 3, 3, 3,
                                                  50, False)
            bbk = [np.where(borders_cc.np == cc) for cc in range(1, ccn)]
            seeds = []
            for cc in range(ccn - 1):
                mi = np.argmax(borders[bbk[cc]])
                seeds.append([bbk[cc][i][mi] for i in range(4)])
            seeds = tuple(np.array(seeds).T)
            borders.fill(0)
            borders[borders_cc.np != 0] = 1
            borders[seeds] = range(2, ccn + 1)
            aimsalgo.AimsDistanceFrontPropagation(borders, 1, 0, 3, 3, 3,
                                                  50, False)
            seeds2 = []
            for cc in range(ccn - 1):
                mi = np.argmax(borders[bbk[cc]])
                seeds2.append([bbk[cc][i][mi] for i in range(4)])
            seeds2 = tuple(np.array(seeds2).T)
            borders.fill(0)
            borders[borders_cc.np != 0] = 1
            borders[seeds] = -103  # immportals value
            borders[seeds2] = -103  # immportals value

            tmp = tempfile.mkstemp(prefix='deep_folding_', suffix='.nii')
            os.close(tmp[0])
            tmp2 = tempfile.mkstemp(prefix='deep_folding_sk_', suffix='.nii')
            os.close(tmp2[0])
            tmps = [tmp[1], tmp2[1]]

            try:
                aims.write(borders, tmp[1])
                # aims.write(borders, '/tmp/borders.nii')  # debug
                cmd = ['VipSkeleton', '-i', tmp[1], '-so', tmp2[1], '-fv', 'n',
                       '-sk', 's', '-p', '0', '-c', 'n', '-k']
                subprocess.check_call(cmd)
                borders = aims.read(tmp2[1])
                # skeleton of resampled with immortals
                sk_in = aims.Volume(resampled)
                # aims.write(sk_in, '/tmp/borders_sk.nii')  # debug
                sk_in[sk_in.np != 0] = 1
                sk_in[borders.np != 0] = -103  # immportals value
                del borders
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
            irepl = {v: k for k, v in repl.items()}
            virepl = {v: k for k, v in repl_cc.items()}
            icc = {}
            for k, vv in cc_per_v.items():
                for v in vv:
                    icc[v] = k
            irepl.update({k: icc[v] for k, v in virepl.items()})
            replacer = getattr(
                aims, 'Replacer_{}'.format(aims.typeCode(resampled.np.dtype)))
            replacer.replace(resampled, resampled, irepl)

        if redo_classif and do_skel:
            th = aims.Volume(resampled)
            th[th.np != 0] = 1
            tc = aimsalgo.TopologicalClassifier_Volume_S16()
            topo = tc.doit(th)
            del th
            # replace values changed between aims/vip topological values (see
            # the output of the command "VipTopoClassifMeaning -a") and
            # deep_folding value for external junction, 35, which are now
            # 30 or 80 (bottom or junction).
            resampled[resampled.np != 35] = 0
            resampled2 = aims.Volume(resampled)
            # dilate value 35 1 voxel in each direction
            resampled[:-1, :, :, :] += resampled[1:, :, :, :]
            resampled[1:, :, :, :] += resampled2[:-1, :, :, :]
            resampled[:, :-1, :, :] += resampled2[:, 1:, :, :]
            resampled[:, 1:, :, :] += resampled2[:, :-1, :, :]
            resampled[:, :, :-1, :] += resampled2[:, :, 1:, :]
            resampled[:, :, 1:, :] += resampled2[:, :, :-1, :]
            del resampled2
            # intersect
            topo[np.isin(topo, (30, 80)) & (resampled.np != 0)] = 35
            resampled = topo
            del topo

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
                    resampled[c[0], c[1], c[2]] = values[i]

            log.debug("Time for value {} ({} voxels): {}s".format(
                v, np.where(vol_dt == v)[0].shape[0], time() - tic))
            log.debug("\t{}s to create the bucket\n\t{}s to resample bucket\n"
                      "\t{}s to assign values".format(t_bck, t_rs, time() - toc))
            tic = time()

    return resampled
