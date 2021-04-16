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
	ref_dir = os.path.join(os.getcwd(), 'data/reference/bounding_box')
	ref_dir = os.path.abspath(ref_dir)
	print("ref_dir = " + ref_dir)

	# Defines the target directory
	tgt_dir = os.path.join(os.getcwd(), 'data/target/bounding_box')
	tgt_dir = os.path.abspath(tgt_dir)

	# Gets sulcus name
	sulcus = 'S.T.s.ter.asc.ant._left'
	side = 'L'
	hemisphere = 'Right' if side == 'R' else 'Left'

	# Determines the bounding box around the sulcus
	bounding_box.bounding_box(src_dir=src_dir, tgt_dir=tgt_dir,
							  sulcus=sulcus, side=side,
							  number_subjects=bounding_box._ALL_SUBJECTS)

	# Selected keys to test
	selected_keys = ['bbmin_voxel', 'bbmax_voxel',
					 'bbmin_AIMS_Talairach', 'bbmin_AIMS_Talairach']

	# Gets and read the first reference file
	ref_dir_side = os.path.join(ref_dir, hemisphere)
	ref_file = os.listdir(ref_dir_side)[0]
	print "ref_file = ", ref_file, '\n'
	with open(os.path.join(ref_dir_side, ref_file), 'r') as f:
		data_ref = json.load(f)
		print(json.dumps(data_ref, sort_keys=True, indent=4))
		box_ref = {k: data_ref[k] for k in selected_keys}

	# Gets and read the second reference file
	tgt_dir_side = os.path.join(tgt_dir, hemisphere)
	tgt_file = os.listdir(tgt_dir_side)[0]
	print"tgt_file = ", tgt_file, '\n'
	with open(os.path.join(tgt_dir_side, tgt_file), 'r') as f:
		data_target = json.load(f)
		print(json.dumps(data_target, sort_keys=True, indent=4))
		box_target = {k: data_ref[k] for k in selected_keys}

	assert box_target == box_ref
