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
