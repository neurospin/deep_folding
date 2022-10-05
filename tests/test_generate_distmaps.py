import os
import numpy as np
from soma import aims
from deep_folding.brainvisa import generate_distmaps
from deep_folding.brainvisa.utils.constants import _ALL_SUBJECTS

distmaps_dir = 'data/test'

if os.path.isdir('/neurospin/dico/data/deep_folding/current/datasets/hcp/skeletons/raw/'):
    src_dir = '/neurospin/dico/data/deep_folding/current/datasets/hcp/skeletons/raw/'
    tgt_dir = '/tmp/'
    ref_dir = '/neurospin/dico/data/deep_folding/test/distmap'
else:
    src_dir = '/nfs/neurospin/dico/data/deep_folding/current/datasets/hcp/skeletons/raw/'
    tgt_dir = '/tmp/'
    ref_dir = '/nfs/neurospin/dico/data/deep_folding/test/distmap'


def equal_distmaps(distmap_ref, distmap_target):
    """Returns True if distmap1 and distmap2 are identical
    """
    equal_distmap = np.array_equal(distmap_ref, distmap_target)
    return equal_distmap


def test_generate_distmaps_help():
    args = "--help"
    argv = args.split(' ')
    generate_distmaps.main(argv)


def test_generate_distmaps_n_0():
    """Tests the function when number of subjects is 0"""
    generate_distmaps.generate_distmaps(
        distmaps_dir=distmaps_dir,
        number_subjects=0)


def test_generate_distmaps():
    """Test distmap generation
    """
    generate_distmaps.generate_distmaps(src_dir=src_dir,
                        distmaps_dir=tgt_dir,
                        side='R',
                        parallel=False,
                        resampled_skel=False,
                        number_subjects=1)

    distmap_target = aims.read(os.path.join(
        tgt_dir, 'R', 'Rdistmap_generated_126426.nii.gz')).arraydata()

    distmap_ref = aims.read(os.path.join(
        ref_dir, 'Rdistmap_generated_126426.nii.gz')).arraydata()

    equal_dist = equal_distmaps(distmap_ref, distmap_target)

    assert equal_dist
