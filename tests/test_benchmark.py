from soma import aims
import numpy as np
import pandas as pd
import os
from deep_folding.brainvisa.benchmark_generation_distmap import Benchmark

# if os.path.isdir('/neurospin/'):
#     mask_dir='/neurospin/dico/data/deep_folding/current/mask/1mm/'
# else:
#     mask_dir = '/nfs/neurospin/dico/data/deep_folding/current/mask/1mm/'

mask_dir = os.path.join(os.getcwd(), 'data/mask/1mm')

def equal_skeletons(skel_ref, skel_target):
    """Returns True if skel1 and skel2 are identical
    """
    equal_skeleton = np.array_equal(skel_ref, skel_target)
    return equal_skeleton


def equal_csv_files(csv1, csv2):
    """Returns True if csv1 and csv2 are identical
    """
    csv1 = pd.read_csv(csv1)
    csv2 = pd.read_csv(csv2)
    equal_csv = csv1.equals(csv2)
    return equal_csv

def are_arrays_almost_equal(arr1, arr2, epsilon, max_number_different_pixels):
	"""Returns True if arrays arr1 and arr2 are almost equal

	arr1 and arr2 are almost equal if at most max_number_different_pixels pixels
	differ by more than epsilon

	Args:
		arr1: first numpy array
		arr2: second numpy array
		epsilon: max allowed difference between pixels values
		max_number_different_pixels: max allowed different number of pixels

	Returns:
		equal_arrays: True if arrays are almost equal
		number_different_pixels: number of pixels differing by more thanepsilon
    """
	difference = (abs(arr1 - arr2) >= epsilon)
	number_different_pixels = np.count_nonzero(difference)
	equal_arrays = number_different_pixels <= max_number_different_pixels
	return equal_arrays, number_different_pixels

def test_suppr_benchmark():
    """Tests suppr benchmark generation
    """
    src_dir = os.path.join(
        os.getcwd(), 'data/source/unsupervised/ANALYSIS/3T_morphologist')
    tgt_dir = os.path.join(os.getcwd(), 'data/target/benchmark/benchmark1')
    bbox_dir = os.path.join(os.getcwd(), 'data/reference/bbox/')
    sulci_list = ['S.C._right']

    benchmark = Benchmark(1, 'R', 200, sulci_list, data_dir=src_dir,
                          saving_dir=tgt_dir, bbox_dir=bbox_dir,
                          mask_dir=mask_dir)
    print(src_dir)
    subjects_list = ['100206']

    abnormality_test = []
    givers = []

    try:
        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)
    except OSError:
        print("Creation of the directory %s failed" % tgt_dir)
    print("Successfully created the directory %s" % tgt_dir)

    for i, sub in enumerate(subjects_list):
        benchmark.get_simple_surfaces(sub)
        if benchmark.surfaces and len(benchmark.surfaces.keys()) > 0:
            benchmark.generate_skeleton(sub)
            save_sub = benchmark.delete_ss(sub)
            abnormality_test.append(save_sub)
            benchmark.save_file(save_sub)
            benchmark.save_lists(abnormality_test, givers, subjects_list)

    skel_target = aims.read(os.path.join(
        tgt_dir, 'modified_skeleton_100206.nii.gz')).arraydata()

    ref_dir = os.path.join(os.getcwd(), 'data/reference/benchmark')
    skel_ref = aims.read(os.path.join(
        ref_dir, 'skeleton_100206_suppr.nii.gz')).arraydata()

    # equal_skel = equal_skeletons(skel_ref, skel_target)
    equal_skel, nb_different_pixels = are_arrays_almost_equal(skel_ref, skel_target, 1, 0)
    print(f"skel_target: {np.unique(skel_target, return_counts=True)}")
    print(f"skel_ref: {np.unique(skel_ref, return_counts=True)}")
    print(f"nb of different pixels: {nb_different_pixels}")
    assert equal_skel

    tgt_csv = os.path.join(
        os.getcwd(),
        'data/target/benchmark/benchmark1/abnormality_test.csv')
    ref_csv = os.path.join(
        os.getcwd(),
        'data/reference/benchmark/abnormality_test_suppr.csv')
    equal_csv = equal_csv_files(tgt_csv, ref_csv)
    assert equal_csv

if __name__ == '__main__':
    test_suppr_benchmark()
