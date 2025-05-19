from deep_folding.brainvisa import generate_crops
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

crop_dir = 'data/test'

def test_generate_crops_help():
    args = "--help"
    argv = args.split(' ')
    generate_crops.main(argv)

def test_generate_crops_n_0():
    """Tests the function when number of subjects is 0"""
    generate_crops.generate_crops(crop_dir=crop_dir, nb_subjects=0)
