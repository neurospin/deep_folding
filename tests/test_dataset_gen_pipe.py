import os
import json

from deep_folding.anatomist_tools import bounding_box


def test_bounding_box():
	"""Tests if the bounding box one one subject gives the expected result.

	The source and the reference are in the data subdirectory
	"""

	# Gets the source directory
	src_dir = os.path.join(os.getcwd(), 'data/source/supervised')
	src_dir = os.path.abspath(src_dir)

	# Gets the reference directory
	ref_dir = os.path.join(os.getcwd(), 'data/reference/bbox')
	ref_dir = os.path.abspath(ref_dir)
	print("ref_dir = " + ref_dir)

	# Defines the target directory
	tgt_dir = os.path.join(os.getcwd(), 'data/target/bbox')
	tgt_dir = os.path.abspath(tgt_dir)

	# Gets normalized SPM file
	norm_dir = os.path.join(os.getcwd(), 'data/source/unsupervised')
	norm_dir = os.path.abspath(norm_dir)
	sub_dir = "ANALYSIS/3T_morphologist/100206/t1mri/default_acquisition"
	file_name = "normalized_SPM_100206.nii"
	image_normalized_spm = os.path.join(norm_dir, sub_dir, file_name)

	# Gets sulcus name
	sulcus = 'S.T.s.ter.asc.ant._left'
	side = 'L'

	# Determines the bounding box around the sulcus
	bounding_box.bounding_box(src_dir=src_dir, tgt_dir=tgt_dir,
							  sulcus=sulcus, side=side,
							  number_subjects=bounding_box._ALL_SUBJECTS,
							  image_normalized_spm=image_normalized_spm)

	# Selected keys to test
	selected_keys = ['bbmin_voxel', 'bbmax_voxel',
					 'bbmin_AIMS_Talairach', 'bbmin_AIMS_Talairach']

	# Gets and read the first reference file
	ref_dir_side = os.path.join(ref_dir, side)
	ref_file = os.listdir(ref_dir_side)[0]
	print("ref_file = ", ref_file, '\n')
	with open(os.path.join(ref_dir_side, ref_file), 'r') as f:
		data_ref = json.load(f)
		print(json.dumps(data_ref, sort_keys=True, indent=4))
		box_ref = {k: data_ref[k] for k in selected_keys}

	# Gets and read the second reference file
	tgt_dir_side = os.path.join(tgt_dir, side)
	tgt_file = os.listdir(tgt_dir_side)[0]
	print("tgt_file = ", tgt_file, '\n')
	with open(os.path.join(tgt_dir_side, tgt_file), 'r') as f:
		data_target = json.load(f)
		print(json.dumps(data_target, sort_keys=True, indent=4))
		box_target = {k: data_ref[k] for k in selected_keys}

	assert box_target == box_ref
