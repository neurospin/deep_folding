from soma import aims
import os
from deep_folding.anatomist_tools import transform


def test_transform():
    """Tests if the transformation of one subject gives the expected result

    The source and the reference are in the data subdirectory
    """

    # Gets the source directory
    src_dir = os.path.join(os.getcwd(), 'data/source/unsupervised')
    src_dir = os.path.abspath(src_dir)

    # Gets the reference directory
    ref_dir = os.path.join(os.getcwd(), 'data/reference/transform')
    ref_dir = os.path.abspath(ref_dir)
    print("ref_dir = " + ref_dir)

    # Defines the target directory
    tgt_dir = os.path.join(os.getcwd(), 'data/target/transform')
    tgt_dir = os.path.abspath(tgt_dir)

    # Performs the actual transform
    transform.transform_to_spm(src_dir=src_dir, tgt_dir=tgt_dir,
                               number_subjects=transform._ALL_SUBJECTS)

    # takes and reads first target file
    tgt_file = os.path.join(tgt_dir, os.listdir(tgt_dir)[0])
    tgt_transfo = aims.read(tgt_file)
    print(tgt_transfo)

    # takes and read first reference file
    ref_file = os.path.join(ref_dir, os.listdir(ref_dir)[0])
    ref_transfo = aims.read(ref_file)
    print(type(ref_transfo))
    print(ref_transfo)

    assert tgt_transfo == ref_transfo
