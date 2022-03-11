_ALL_SUBJECTS = -1


# Default directory in which lies the manually segmented database
_SUPERVISED_SRC_DIR_DEFAULT = "/neurospin/dico/data/bv_databases/human/pclean/all"

# Default directory in which lies the dataset
_SRC_DIR_DEFAULT = "/tgcc/hcp/ANALYSIS/3T_morphologist"

# Default directory to which we write the bounding box results
_BBOX_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/bbox"

# Default directory to which we write the masks
_MASK_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/mask"

# Default directory where we put skeletons
_SKELETON_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/hcp/skeletons/raw"

# Default directory where we put foldlabels
_FOLDLABEL_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/hcp/foldlabels/raw"

# Default directory where we put resampled skeletons
_RESAMPLED_SKELETON_DIR_DEFAULT = "/neurospin/dico/data/deep_folding/test/hcp/skeletons/1mm"

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'R'

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'F.C.M.ant.'

# voxel size:
_VOXEL_SIZE_DEFAULT = 1.0

# junction type 'wide' or 'thin'
_JUNCTION_DEFAULT = 'thin'

# Gives the relative path to the manually labelled graph .arg
# in the supervised database
_PATH_TO_GRAPH_SUPERVISED_DEFAULT = \
    "t1mri/t1/default_analysis/folds/3.3/base2018_manual"

# Gives the relative path to the labelled graph .arg
# in a "morphologist-like" dataset
_PATH_TO_GRAPH_DEFAULT = \
    "t1mri/default_acquisition/default_analysis/folds/3.1/default_session_*"
