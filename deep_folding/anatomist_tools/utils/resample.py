"""
Script from https://github.com/BastienCagna/labelresample.git
"""

import numpy as np
from soma import aims, aimsalgo


def resample(input_image, output_image, transformation=None, output_vs=None, background=11):
    """
        Transform and resample a volume that as discret values

        Parameters
        ----------
        input_image: file
            Path to the input volume (.nii or .nii.gz file)
        transformation: file
            Linear transformation file (.trm file)
        output_vs: tuple
            Output voxel size (default: None, no resampling)
        background: int
            Background value (default: 11)

        Return
        ------
        resampled_vol:
            Transformed and resampled volume
    """
    # Read inputs
    vol = aims.read(input_image)
    vol_dt = vol.__array__()

    if transformation:
        trm = aims.read(transformation)
    else:
        trm = aims.AffineTransformation3d(np.eye(4))
    inv_trm = trm.inverse()

    if output_vs:
        output_vs = np.array(output_vs)

        # New volume dimensions
        resampling_ratio = np.array(vol.header()['voxel_size'][:3]) / output_vs
        orig_dim = vol.header()['volume_dimension'][:3]
        new_dim = list((resampling_ratio * orig_dim).astype(int))
    else:
        output_vs = vol.header()['voxel_size'][:3]
        new_dim = vol.header()['volume_dimension'][:3]

    # Transform the background
    # Using the inverse is more straightforward and supports non-linear
    # transforms
    resampled = aims.Volume(new_dim, dtype=vol_dt.dtype)
    resampled.header()['voxel_size'] = output_vs
    # 0 order (nearest neightbours) resampling
    resampler = aimsalgo.ResamplerFactory(vol).getResampler(0)
    resampler.setDefaultValue(background)
    resampler.setRef(vol)
    resampler.resample_inv(vol, inv_trm, 0, resampled)
    resampled_dt = np.asarray(resampled)

    # Create one bucket by value (except background)
    # FIXME: Create several buckets because I didn't understood how to add
    #  several bucket to a BucketMap
    values = np.unique(vol_dt[vol_dt != background])
    # TODO: add pissiblity to order values by priority
    for i, v in enumerate(values):
        bck = aims.BucketMap_VOID()
        bck.setSizeXYZT(*vol.header()['voxel_size'][:3], 1.)
        bk0 = bck[0]
        for p in np.vstack(np.where(vol_dt == v)[:3]).T:
            bk0[list(p)] = v

        bck2 = aimsalgo.resampleBucket(bck, trm, inv_trm, output_vs)

        # FIXME: Could not assign the correct value with the converter.
        # Using the converter, the new_dim must incremented
        # conv = aims.Converter(intype=bck2, outtype=aims.AimsData(vol))
        # conv.convert(bck2, resampled)
        # Use a for loop instead:
        for p in bck2[0].keys():
            c = p.list()
            if c[0] < new_dim[0] and c[1] < new_dim[1] and c[2] < new_dim[2]:
                resampled_dt[c[0], c[1], c[2]] = values[i]

    aims.write(resampled, output_image)
    return resampled


if __name__ == '__main__':
    dir = '/neurospin/dico/lguillon/skeleton/resampling_nn_2mm/Rcrops/'
    input_image = dir + 'Rskeleton_100307.nii.gz'
    tr = '/neurospin/dico/deep_folding_data/data/transform/natif_to_template_spm_100307.trm'
    #input_image = dir + '129533_normalized.nii.gz'
    resampled = resample(input_image, output_vs=(2,2,2),transformation=tr)
    aims.write(resampled, dir + 'test_5_trm.nii.gz')
