from deep_folding.brainvisa import compute_mask
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

mask_dir = 'data/test'

def test_compute_mask_help():
    args = "--help"
    argv = args.split(' ')
    compute_mask.main(argv)

def test_compute_mask_n_0():
    """Tests the function when number of subjects is 0"""
    compute_mask.compute_mask(mask_dir=mask_dir, number_subjects=0)
