#!python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

"""Puts together crops of different datasets."""

import os
import json
import numpy as np
import pandas as pd
import shutil

from deep_folding.brainvisa.utils.folder import create_folder


regions = ['S.Or.']
datasets = ['bsnip1', 'candi', 'cnp', 'schizconnect-vip-prague']
target = 'schiz'
resolution = "2mm"

datasets_folder = "/neurospin/dico/data/deep_folding/current/datasets"
target_folder = f"{datasets_folder}/{target}"


def concat_npy(result_npy, new_npy_file):
    new_npy = np.load(new_npy_file)
    if result_npy.size:
        if result_npy.shape[1:] != new_npy.shape[1:]:
            raise ValueError(
                "Numpy shapes do not match: "
                f"Concatenated numpy has individual shape {result_npy.shape[1:]}\n"
                f"Shape coming from file {new_npy_file} has shape {new_npy.shape[1:]}")
        result_npy = np.concatenate([result_npy, new_npy], axis=0)
    else:
        result_npy = new_npy
    return result_npy


def concat_csv(result_csv, new_csv_file):
    new_csv = pd.read_csv(new_csv_file)
    result_csv = pd.concat([result_csv, new_csv], axis=0, ignore_index=True)
    return result_csv


def check_if_same_position(s, f, d):
    assert (s.shape == f.shape), (
        f"Skeleton and foldlabel of different shapes: {s.shape} != {f.shape}")
    assert (s.shape == d.shape), (
        f"Skeleton and distbottom of different shapes: {s.shape} != {d.shape}")
    assert (s[d==32501].sum() == 0), (
        f"Skeleton and distbottom with different non-zero positions: "
        f"{(s[d==32501]!=0).sum()} different non-zero positions")
    assert ((d[s==0]!=32501).sum() == 0), (
        f"Skeleton and distbottom with different non-zero positions: "
        f"{(d[s==0]!=32501).sum()} different non-zero positions")
    assert (f[s==0].sum() == 0), (
        f"Foldlabel and skeleton arrays with different non-zero positions: "
        f"{(f[s==0]!=0).sum()} different non-zero positions")
    assert (s[f==0].sum() == 0), (
        f"Foldlabel and skeleton arrays with different non-zero positions: "
        f"{(s[f==0]!=0).sum()} different non-zero positions")


def check_if_equal(list_dataframes):
    df0 = list_dataframes[0]
    for df in list_dataframes[1:]:
        assert (df.equals(df0)), "List of subjects are not equal"


def check_if_same_dim(arr, df):
    assert (arr.shape[0] == len(df)), "Number of subjects differs between numpy array and csv"


def save_to_numpy(npy_file, arr):
    np.save(npy_file, arr)
    reloaded = np.load(npy_file)
    assert np.array_equal(arr, reloaded), "Array on disk differs from computed array"


def save_to_csv(csv_file, df):
    df.to_csv(csv_file, index=False)
    reloaded = pd.read_csv(csv_file)
    assert df.equals(reloaded), "csv file on disk differs from computed csv dataframe"


for region in regions:

    # Prepares output directories
    target_region_folder = f"{target_folder}/crops/{resolution}/{region}/mask"
    shutil.rmtree(target_region_folder, ignore_errors=True)
    create_folder(target_region_folder)

    # Initializes target numpy arrays
    Rskeleton_npy = np.array([])
    Rlabel_npy = np.array([])
    Rdistbottom_npy = np.array([])
    Lskeleton_npy = np.array([])
    Llabel_npy = np.array([])
    Ldistbottom_npy = np.array([])

    # Initializes target dataFrames
    Rskeleton_subject_csv = pd.DataFrame()
    Rlabel_subject_csv = pd.DataFrame()
    Rdistbottom_subject_csv = pd.DataFrame()
    Lskeleton_subject_csv = pd.DataFrame()
    Llabel_subject_csv = pd.DataFrame()
    Ldistbottom_subject_csv = pd.DataFrame()

    for dataset in datasets:
        # Goes to region folder
        src_folder = f"{datasets_folder}/{dataset}/crops/{resolution}/{region}/mask"

        # Concatenates numpy
        Rskeleton_npy = concat_npy(Rskeleton_npy, f"{src_folder}/Rskeleton.npy")
        Rlabel_npy = concat_npy(Rlabel_npy, f"{src_folder}/Rlabel.npy")
        Rdistbottom_npy = concat_npy(Rdistbottom_npy, f"{src_folder}/Rdistbottom.npy")
        Lskeleton_npy = concat_npy(Lskeleton_npy, f"{src_folder}/Lskeleton.npy")
        Llabel_npy = concat_npy(Llabel_npy, f"{src_folder}/Llabel.npy")
        Ldistbottom_npy = concat_npy(Ldistbottom_npy, f"{src_folder}/Ldistbottom.npy")

        # Concatenates csv
        Rskeleton_subject_csv = concat_csv(Rskeleton_subject_csv, f"{src_folder}/Rskeleton_subject.csv")
        Rlabel_subject_csv = concat_csv(Rlabel_subject_csv, f"{src_folder}/Rlabel_subject.csv")
        Rdistbottom_subject_csv = concat_csv(Rdistbottom_subject_csv, f"{src_folder}/Rdistbottom_subject.csv")
        Lskeleton_subject_csv = concat_csv(Lskeleton_subject_csv, f"{src_folder}/Lskeleton_subject.csv")
        Llabel_subject_csv = concat_csv(Llabel_subject_csv, f"{src_folder}/Llabel_subject.csv")
        Ldistbottom_subject_csv = concat_csv(Ldistbottom_subject_csv, f"{src_folder}/Ldistbottom_subject.csv")

    # Checks numpy
    check_if_same_position(Rskeleton_npy, Rlabel_npy, Rdistbottom_npy)
    check_if_same_position(Lskeleton_npy, Llabel_npy, Ldistbottom_npy)

    # Checks dataframes
    check_if_equal([Rskeleton_subject_csv, Rlabel_subject_csv, Rdistbottom_subject_csv])
    check_if_equal([Lskeleton_subject_csv, Llabel_subject_csv, Ldistbottom_subject_csv])

    # Cross-checks numpy and dataframes
    check_if_same_dim(Rskeleton_npy, Rskeleton_subject_csv)
    check_if_same_dim(Rlabel_npy, Rlabel_subject_csv)
    check_if_same_dim(Rdistbottom_npy, Rdistbottom_subject_csv)
    check_if_same_dim(Lskeleton_npy, Lskeleton_subject_csv)
    check_if_same_dim(Llabel_npy, Llabel_subject_csv)
    check_if_same_dim(Ldistbottom_npy, Ldistbottom_subject_csv)

    # Saves numpy
    save_to_numpy(f"{target_region_folder}/Rskeleton.npy", Rskeleton_npy)
    save_to_numpy(f"{target_region_folder}/Rlabel.npy", Rlabel_npy)
    save_to_numpy(f"{target_region_folder}/Rdistbottom.npy", Rdistbottom_npy)
    save_to_numpy(f"{target_region_folder}/Lskeleton.npy", Lskeleton_npy)
    save_to_numpy(f"{target_region_folder}/Llabel.npy", Llabel_npy)
    save_to_numpy(f"{target_region_folder}/Ldistbottom.npy", Ldistbottom_npy)

    # Saves csv
    save_to_csv(f"{target_region_folder}/Rskeleton_subject.csv", Rskeleton_subject_csv)
    save_to_csv(f"{target_region_folder}/Rlabel_subject.csv", Rlabel_subject_csv)
    save_to_csv(f"{target_region_folder}/Rdistbottom_subject.csv", Rdistbottom_subject_csv)
    save_to_csv(f"{target_region_folder}/Lskeleton_subject.csv", Lskeleton_subject_csv)
    save_to_csv(f"{target_region_folder}/Llabel_subject.csv", Llabel_subject_csv)
    save_to_csv(f"{target_region_folder}/Ldistbottom_subject.csv", Ldistbottom_subject_csv)
