from soma import aims
from deep_folding.anatomist_tools.utils.resample import resample
import numpy as np
from time import time


skeleton = "/home/bastien/data/archi/t1-1mm-1/025/t1mri/default_acquisition/" \
           "default_analysis/segmentation/Lskeleton_025.nii.gz"
rs_skeleton = "/var/tmp/resampled_Lskeleton_025.nii.gz"
ors_skeleton = "/var/tmp/ordered_resampled_Lskeleton_025.nii.gz"


vol = aims.read(skeleton)
print("Input:", vol.header()['volume_dimension'], vol.header()['voxel_size'])

print("\nResampling (without specific order)")
tic = time()
rvol = resample(skeleton, None, (2, 2, 2))
print('Resampling took {:.0f} seconds'.format(time()-tic))
aims.write(rvol, rs_skeleton)
print("Output:", rvol.header()['volume_dimension'], rvol.header()['voxel_size'])


print("\nResampling (with specific order)")
tic = time()
orvol = resample(skeleton, None, (2, 2, 2), 11, [80, 60, 11, 30, 40, 70, 10, 0])
print('Resampling took {:.0f} seconds'.format(time()-tic))
aims.write(orvol, ors_skeleton)
print("Output:", orvol.header()['volume_dimension'],
      orvol.header()['voxel_size'])


print("\nLabels percentages:")
dt = np.asarray(vol)
n = dt.shape[0] * dt.shape[1] * dt.shape[2]
rdt = np.asarray(rvol)
rn = rdt.shape[0] * rdt.shape[1] * rdt.shape[2]
ordt = np.asarray(orvol)
orn = ordt.shape[0] * ordt.shape[1] * ordt.shape[2]
for v in np.unique(dt):
    print("{}:\t{:.02f}%\t{:0.02f}%\t{:0.02f}%".format(
        v, 100*np.sum(dt == v)/n, 100*np.sum(rdt == v)/rn,
        100*np.sum(ordt == v)/orn))
