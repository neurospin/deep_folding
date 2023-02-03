import os
from soma import aims
import numpy as np
import random
import dico_toolbox as dtx
from deep_folding.brainvisa import generate_foldlabels
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

foldlabel_dir = 'data/test'

def test_generate_foldlabels_help():
    args = "--help"
    argv = args.split(' ')
    generate_foldlabels.main(argv)


def test_generate_foldlabels_n_0():
    """Tests the function when number of subjects is 0"""
    generate_foldlabels.generate_foldlabels(
        foldlabel_dir=foldlabel_dir,
        number_subjects=0)


def equal_buckets(bck1, bck2):
    """Returns True if the two buckets bck1 and bck2 are identical
    """
    bck1 = aims.read(bck1)
    bck2 = aims.read(bck2)
    arr1 = dtx.convert.bucketMAP_aims_to_ndarray(bck1)
    arr2 = dtx.convert.bucketMAP_aims_to_ndarray(bck2)
    return np.array_equal(arr1, arr2)


def apply_mask(arr, mask):
    """Applies a ROI mask on an hemisphere
    arr: array on which apply the mask
    mask: name of the mask to apply
    """
    #mask_dir = '/home/lg261972/Documents/deep_folding/data/reference/mask/R'
    mask_dir = os.path.join(os.getcwd(), 'data/reference/mask/R')
    mask = aims.read(os.path.join(mask_dir, mask + '.nii.gz'))
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    arr[mask==0] = 0

    return arr


def test_generate_foldlabels_value_correspondance():
    """ Tests the correspondance of the values of simple surfaces, bottom and
        junction. """
    #data_dir = os.path.join(os.getcwd(), 'data/reference/foldlabel/')
    tgt_dir = os.path.join(os.getcwd(), 'data/target/foldlabel')
    list_masks = ['S.C._right', 'F.C.M.ant._right', 'S.Or._right']
    random.seed(0)

    for mask in list_masks:
        #data_dir = '/home/lg261972/Documents/deep_folding/data/reference/foldlabel'
        data_dir = os.path.join(os.getcwd(), 'data/reference/foldlabel')
        foldlabel = aims.read(os.path.join(data_dir,
                            'Rresampled_foldlabel_129533.nii.gz'))
        skeleton = aims.read(os.path.join(data_dir,
                            'Rresampled_skeleton_129533.nii.gz'))
        foldlabel = apply_mask(foldlabel, mask)
        fold_arr = np.asarray(foldlabel)
        skeleton = apply_mask(skeleton, mask)
        skel_arr = np.asarray(skeleton)

        folds_list = np.unique(foldlabel, return_counts=True)
        folds_dico = {key: value for key, value in zip(folds_list[0], folds_list[1]) if value>=150}

        # We don't take into account the background in the random choice of fold
        folds_dico.pop(0, None)
        fold = random.choice(list(folds_dico.keys()))

        fold_arr[fold_arr==fold] = 9999
        # Selection of associated bottom
        fold_arr[fold_arr==fold + 6000] = 9999
        # Selection of other associated junction
        fold_arr[fold_arr==fold + 5000] = 9999

        ## suppression of chosen folds
        skeleton[fold_arr==9999] = -1
        skel_arr[skel_arr==-1] = 0

        if not os.path.exists(tgt_dir):
            os.makedirs(tgt_dir)

        aims.write(dtx.convert.volume_to_bucketMap_aims(np.squeeze(skeleton)), f"{tgt_dir}/skel_suppr_{mask}.bck")

        equal_buckets(f"{tgt_dir}/skel_suppr_{mask}.bck",
                      f"{data_dir}/skel_suppr_{mask}.bck")


if __name__ == '__main__':
    test_generate_foldlabels_value_correspondance()
