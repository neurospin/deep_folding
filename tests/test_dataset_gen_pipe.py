import os
import glob

import numpy as np

from soma import aims

from deep_folding.anatomist_tools import dataset_gen_pipe

_ALL_SUBJECTS = -1

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

def test_dataset_gen_pipe_linear():
	"""Tests if the dataset generation on one subject gives the expected result.

	The source and the reference are in the data subdirectory
	"""

	# Gets the source directory
	src_dir = os.path.join(os.getcwd(), 'data/source/unsupervised')
	src_dir = os.path.abspath(src_dir)

	# Gets the reference, transform and bbox directory
	ref_dir = os.path.join(os.getcwd(), 'data/reference/data/linear')
	ref_dir = os.path.abspath(ref_dir)
	print("ref_dir = " + ref_dir)
	transform_dir = os.path.join(os.getcwd(), 'data/reference/transform')
	transform_dir = os.path.abspath(transform_dir)
	bbox_dir = os.path.join(os.getcwd(), 'data/reference/bbox')
	bbox_dir = os.path.abspath(bbox_dir)

	# Defines the target directory
	tgt_dir = os.path.join(os.getcwd(), 'data/target/data/linear')
	tgt_dir = os.path.abspath(tgt_dir)

	# Gets sulcus name and hemisphere side
	sulcus = 'S.T.s.ter.asc.ant._left'
	side = 'L'

	# Determines the bounding box around the sulcus
	dataset_gen_pipe.dataset_gen_pipe(
		src_dir=src_dir,
		tgt_dir=tgt_dir,
		transform_dir=transform_dir,
		bbox_dir=bbox_dir,
		list_sulci=sulcus,
		side=side,
		interp='linear',
		number_subjects=_ALL_SUBJECTS)

	# Gets target crop as numpy array
	cropped_target_dir = os.path.join(tgt_dir, side+'crops')
	vol_target_file = glob.glob(cropped_target_dir + '/' + '*.nii.gz')
	vol_target = aims.read(vol_target_file[0])
	arr_target = vol_target.arraydata()

	# Gets reference crop as numpy array
	cropped_ref_dir = os.path.join(ref_dir, side+'crops')
	vol_ref_file = glob.glob(cropped_ref_dir + '/' + '*.nii.gz')
	vol_ref = aims.read(vol_ref_file[0])
	arr_ref = vol_ref.arraydata()

	# Test fails if arrays differ strictly of more than 1 on more than 2 pixels
	equal_arrays, number_different_pixels = \
		are_arrays_almost_equal(arr_ref, arr_target, 1, 2)
	assert equal_arrays

def test_dataset_gen_pipe_nearest():
	"""Tests if the dataset generation on one subject gives the expected result.

	The source and the reference are in the data subdirectory
	"""

	# Gets the source directory
	src_dir = os.path.join(os.getcwd(), 'data/source/unsupervised')
	src_dir = os.path.abspath(src_dir)

	# Gets the reference, transform and bbox directory
	ref_dir = os.path.join(os.getcwd(), 'data/reference/data/nearest')
	ref_dir = os.path.abspath(ref_dir)
	print("ref_dir = " + ref_dir)
	transform_dir = os.path.join(os.getcwd(), 'data/reference/transform')
	transform_dir = os.path.abspath(transform_dir)
	bbox_dir = os.path.join(os.getcwd(), 'data/reference/bbox')
	bbox_dir = os.path.abspath(bbox_dir)

	# Defines the target directory
	tgt_dir = os.path.join(os.getcwd(), 'data/target/data/nearest')
	tgt_dir = os.path.abspath(tgt_dir)

	# Gets sulcus name and hemisphere side
	sulcus = 'S.T.s.ter.asc.ant._left'
	side = 'L'

	# Determines the bounding box around the sulcus
	dataset_gen_pipe.dataset_gen_pipe(
		src_dir=src_dir,
		tgt_dir=tgt_dir,
		transform_dir=transform_dir,
		bbox_dir=bbox_dir,
		list_sulci=sulcus,
		side=side,
		interp='nearest',
		number_subjects=_ALL_SUBJECTS)

	# Gets target crop as numpy array
	cropped_target_dir = os.path.join(tgt_dir, side+'crops')
	vol_target_file = glob.glob(cropped_target_dir + '/' + '*.nii.gz')
	vol_target = aims.read(vol_target_file[0])
	arr_target = vol_target.arraydata()

	# Gets reference crop as numpy array
	cropped_ref_dir = os.path.join(ref_dir, side+'crops')
	vol_ref_file = glob.glob(cropped_ref_dir + '/' + '*.nii.gz')
	vol_ref = aims.read(vol_ref_file[0])
	arr_ref = vol_ref.arraydata()

	# Test fails if arrays differ strictly of more than 1 on more than 2 pixels
	equal_arrays, number_different_pixels = \
		are_arrays_almost_equal(arr_ref, arr_target, 1, 2)
	assert equal_arrays