from deep_folding.brainvisa import generate_ICBM2009c_transforms
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

transform_dir = 'data/test'

def test_generate_ICBM2009c_transforms_help():
    args = "--help"
    argv = args.split(' ')
    generate_ICBM2009c_transforms.main(argv)

def test_generate_ICBM2009c_transforms_n_0():
    """Tests the function when number of subjects is 0"""

    generate_ICBM2009c_transforms.generate_ICBM2009c_transforms(
        transform_dir=transform_dir,
        nb_subjects=0)
