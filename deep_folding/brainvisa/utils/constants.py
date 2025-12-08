_ALL_SUBJECTS = -1


# Default directory in which lies the manually segmented database
_SUPERVISED_SRC_DIR_DEFAULT = \
    "/neurospin/dico/data/bv_databases/human/manually_labeled/pclean/all"

# Gives the relative path to the manually labelled graph .arg
# in the supervised database
_PATH_TO_GRAPH_SUPERVISED_DEFAULT = \
    "t1mri/t1/default_analysis/folds/3.3/base2018_manual"

# Default directory in which lies the dataset
# It points to HCP directory
_SRC_DIR_DEFAULT = \
    "/neurospin/dico/data/bv_databases/human/not_labeled/hcp/hcp"

# Gives the relative path to the labelled graph .arg
# in a "morphologist-like" dataset
_PATH_TO_GRAPH_DEFAULT = \
    "t1mri/BL/default_analysis/folds/3.1"

# Gives the relative path to the labelled graph .arg
# in a "morphologist-like" dataset
_PATH_TO_SKELETON_WITH_HULL_DEFAULT = \
    "t1mri/BL/default_analysis/segmentation"

# Default directory to which we write the bounding box
_BBOX_DIR_DEFAULT = "test/bbox/1mm"

# Default directory to which we write the masks
_MASK_DIR_DEFAULT = "test/mask/1mm"

# Default directory to which we write the masks
_INPUT_TYPE_DEFAULT = "skeleton"

# Default directory where we put skeletons
_SKELETON_DIR_DEFAULT = "test/datasets/hcp/skeletons/raw"

# Default directory where we put extremities
_EXTREMITIES_DIR_DEFAULT = "test/datasets/hcp/extremities/raw"

# Default directory where we put foldlabels
_FOLDLABEL_DIR_DEFAULT = "test/datasets/hcp/foldlabels/raw"

# Default directory where we put distmaps
_DISTMAPS_DIR_DEFAULT = "test/datasets/hcp/distmaps/raw"

# Default directory where we put transform files to ICBM2009c
_TRANSFORM_DIR_DEFAULT = "test/datasets/hcp/transforms"

# Default directory where we put resampled skeletons
_RESAMPLED_SKELETON_DIR_DEFAULT = "test/datasets/hcp/skeletons/1mm"

# Default directory where we put resampled foldlabels
_RESAMPLED_FOLDLABEL_DIR_DEFAULT = "test/datasets/hcp/foldlabels/1mm"

# Default directory where we put resampled foldlabels
_RESAMPLED_EXTREMITIES_DIR_DEFAULT = "test/datasets/hcp/extremities/1mm"

# Default directory where we put crops
_CROP_DIR_DEFAULT = "test/datasets/hcp/crops/CINGULATE/mask/1mm"

# Dilation size
_DILATION_DEFAULT = 5.0

# Threshold value
_THRESHOLD_DEFAULT = 0.0

# hemisphere 'L' or 'R'
_SIDE_DEFAULT = 'R'

# sulcus to encompass:
# its name depends on the hemisphere side
_SULCUS_DEFAULT = 'F.C.M.ant.'

# voxel size:
_VOXEL_SIZE_DEFAULT = 1.0

# junction type 'wide' or 'thin'
_JUNCTION_DEFAULT = 'thin'

# cropping default for generating crops
_CROPPING_TYPE_DEFAULT = 'mask'  # crops according to a mask by default

# Combines sulci using ordering (if true, ordering matters)
_COMBINE_TYPE_DEFAULT = False

# Whether applying mask or not on cropped file
_NO_MASK_DEFAULT = False

# path to the brain regions' json
_BRAIN_REGION_JSON = \
    '/neurospin/dico/data/deep_folding/current/sulci_regions_overlap.json'

# path to the qc csv
_QC_PATH_DEFAULT = ''

# number of minimal jobs
_NB_JOBS_DEFAULT = 1