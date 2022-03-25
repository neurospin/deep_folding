from deep_folding.brainvisa import generate_distmaps
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

distmaps_dir = 'data/test'

def test_generate_distmaps_help():
    args = "--help"
    argv = args.split(' ')
    generate_distmaps.main(argv)


def test_generate_distmaps_n_0():
    """Tests the function when number of subjects is 0"""
    generate_distmaps.generate_distmaps(
        distmaps_dir=distmaps_dir,
        number_subjects=0)
