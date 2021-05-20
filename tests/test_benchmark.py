from soma import aims
import numpy as np
from glob import glob
import random
import pandas as pd
import os
import deep_folding.anatomist_tools.utils.load_bbox
from deep_folding.anatomist_tools.benchmark_generation import Benchmark


def equal_skeletons(skel_ref, skel_target):
    """Returns True if skel1 and skel2 are identical
    """
    equal_skeleton = np.array_equal(skel_ref, skel_target)
    return equal_skeleton


def equal_csv_files(csv1, csv2):
    """Returns True if csv1 and csv2 are identical
    """
    csv1 = pd.read_csv(csv1)
    csv2 = pd.read_csv(csv2)
    equal_csv = csv1.equals(csv2)
    return equal_csv


def test_suppr_benchmark():
    """Tests suppr benchmark generation
    """
    src_dir = os.path.join(os.getcwd(), 'data/source/unsupervised/ANALYSIS/3T_morphologist/')
    tgt_dir = os.path.join(os.getcwd(), 'data/target/benchmark')
    bbox_dir = os.path.join(os.getcwd(), 'data/reference/bbox/')
    sulci_list=['S.T.s.ter.asc.post._right', 'S.T.s.ter.asc.ant._right']
    
    benchmark = Benchmark(1, 'R', 500, sulci_list, data_dir=src_dir,
                          saving_dir=tgt_dir, bbox_dir=bbox_dir)
    print(src_dir)
    subjects_list = ['100206']

    abnormality_test = []
    givers = []

    tgt_dir = os.path.join(os.getcwd(), 'data/target/benchmark/benchmark1/')
    try:
        os.makedirs(tgt_dir)
    except OSError:
        print ("Creation of the directory %s failed" % tgt_dir)
    else:
        print ("Successfully created the directory %s" % tgt_dir)
        
    for i, sub in enumerate(subjects_list):
        benchmark.get_simple_surfaces(sub)
        if benchmark.surfaces and len(benchmark.surfaces.keys()) > 0:
            save_sub = benchmark.delete_ss(sub)
            abnormality_test.append(save_sub)
            benchmark.save_file(save_sub)
            benchmark.save_lists(abnormality_test, givers, subjects_list)


    skel_target = aims.read(os.path.join(tgt_dir, 'output_skeleton_100206.nii.gz')).arraydata()

    ref_dir = os.path.join(os.getcwd(), 'data/reference/benchmark')
    skel_ref = aims.read(os.path.join(ref_dir, 'skeleton_100206_suppr.nii.gz')).arraydata()

    equal_skel = equal_skeletons(skel_ref, skel_target)
    assert equal_skel

    tgt_csv = os.path.join(os.getcwd(), 'data/target/benchmark/benchmark1/abnormality_test.csv')
    ref_csv = os.path.join(os.getcwd(), 'data/reference/benchmark/abnormality_test_suppr.csv')
    equal_csv = equal_csv_files(tgt_csv, ref_csv)
    assert equal_csv


def test_add_benchmark():
    """Tests add ss benchmark generation
    """
    src_dir = os.path.join(os.getcwd(), 'data/source/unsupervised/ANALYSIS/3T_morphologist/')
    tgt_dir = os.path.join(os.getcwd(), 'data/target/benchmark')
    bbox_dir = os.path.join(os.getcwd(), 'data/reference/bbox/')
    sulci_list=['S.T.s.ter.asc.post._right', 'S.T.s.ter.asc.ant._right']
    benchmark = Benchmark(2, 'R', 1000, sulci_list, data_dir=src_dir,
                          saving_dir=tgt_dir, bbox_dir=bbox_dir)
    print(src_dir)
    subjects_list = ['100206', '100307']

    abnormality_test = []
    givers = []

    tgt_dir = os.path.join(os.getcwd(), 'data/target/benchmark/benchmark2/')
    try:
        os.makedirs(tgt_dir)
    except OSError:
        print ("Creation of the directory %s failed" % tgt_dir)
    else:
        print ("Successfully created the directory %s" % tgt_dir)
        
    for i, sub in enumerate(subjects_list):
        benchmark.get_simple_surfaces(sub)
        if benchmark.surfaces and len(benchmark.surfaces.keys()) > 0:
            save_sub = benchmark.add_ss(subjects_list, i)
            givers.append(sub)
            # Addition of modified graph to abnormality_test set
            abnormality_test.append(save_sub)
            benchmark.save_file(save_sub)
            benchmark.save_lists(abnormality_test, givers, subjects_list)
            if len(abnormality_test) == 1:
                break

    skel_target = aims.read(os.path.join(tgt_dir, 'output_skeleton_100307.nii.gz')).arraydata()

    ref_dir = os.path.join(os.getcwd(), 'data/reference/benchmark')
    skel_ref = aims.read(os.path.join(ref_dir, 'skeleton_100307_add.nii.gz')).arraydata()

    equal_skel = equal_skeletons(skel_ref, skel_target)
    assert equal_skel

    tgt_csv = os.path.join(os.getcwd(), 'data/target/benchmark/benchmark2/abnormality_test.csv')
    ref_csv = os.path.join(os.getcwd(), 'data/reference/benchmark/abnormality_test_add.csv')
    equal_csv = equal_csv_files(tgt_csv, ref_csv)
    assert equal_csv

    tgt_csv = os.path.join(os.getcwd(), 'data/target/benchmark/benchmark2/givers.csv')
    ref_csv = os.path.join(os.getcwd(), 'data/reference/benchmark/givers.csv')
    equal_csv = equal_csv_files(tgt_csv, ref_csv)
    assert equal_csv


#test_suppr_benchmark()
#test_add_benchmark()
