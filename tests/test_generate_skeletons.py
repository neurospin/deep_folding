from deep_folding.brainvisa import generate_skeletons
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

skeleton_dir = 'data/test'

def test_generate_skeletons_help():
    args = "--help"
    argv = args.split(' ')
    generate_skeletons.main(argv)


def test_generate_skeletons_n_0():
    """Tests the function when number of subjects is 0"""
    generate_skeletons.generate_skeletons(
        skeleton_dir=skeleton_dir,
        nb_subjects=0)
