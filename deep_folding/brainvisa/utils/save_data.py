"""
Scripts that enables to create a dataframe of numpy arrays from .nii.gz or .nii
images.
"""
from __future__ import division

import os
import re

import numpy as np
import pandas as pd
from soma import aims

from deep_folding.config.logs import set_file_logger

# Defines logger
log = set_file_logger(__file__)

def is_file_nii(filename):
    """Tests if file is nii file

    Args:
        filename: string giving file name with full path

    Returns:
        is_file_nii: boolean stating if file is nii file
    """
    is_file_nii = os.path.isfile(filename)\
        and '.nii' in filename \
        and '.minf' not in filename
    return is_file_nii


def save_to_pickle(cropped_dir, tgt_dir=None, file_basename=None):
    """
    Creates a dataframe of data with a column for each subject and associated
    np.array. Saved these this dataframe to pkl format on the target
    directory

    Args:
        cropped_dir: directory containing cropped images
        tgt_dir: directory where to save the pickle file
        file_basename: final file name = file_basename.pkl
    """

    data_dict = dict()

    log.info("Now generating pickle file...")
    log.debug(f"cropped_dir = {cropped_dir}")

    for filename in os.listdir(cropped_dir):
        file_nii = os.path.join(cropped_dir, filename)
        if is_file_nii(file_nii):
            aimsvol = aims.read(file_nii)
            sample = np.asarray(aimsvol)
            subject = re.search('([ae\\d]{5,6})', file_nii).group(1)
            data_dict[subject] = [sample]

    dataframe = pd.DataFrame.from_dict(data_dict)

    file_pickle_basename = file_basename + '.pkl'
    file_pickle = os.path.join(tgt_dir, file_pickle_basename)
    dataframe.to_pickle(file_pickle)


def save_to_numpy(cropped_dir, tgt_dir=None, file_basename=None):
    """
    Creates a dataframe of data with a column for each subject and associated
    np.array. Saved these this dataframe to pkl format on the target
    directory

    Args:
        cropped_dir: directory containing cropped images
        tgt_dir: directory where to save the pickle file
        file_basename: final file name = file_basename.pkl
    """
    list_sample_id = []
    list_sample_file = []

    log.info("Now generating numpy array...")
    log.debug(f"cropped_dir = {cropped_dir}")
    #list_sub = np.load('/neurospin/dico/lguillon/distmap/data/test_list.csv')
    #list_sub = pd.read_csv('/neurospin/dico/lguillon/distmap/data/test_list.csv').subjects.astype(str).values
    for filename in os.listdir(cropped_dir):
        #print(list_sub)
        #for filename in list_sub:
        print(filename)
        file_nii = os.path.join(cropped_dir, filename)
        #file_nii = os.path.join(cropped_dir, filename+ '_cropped_skeleton.nii.gz')
        #file_nii = os.path.join(cropped_dir, 'Rdistmap_generated_' + filename+ '.nii.gz')
        #file_nii = os.path.join(cropped_dir, 'Ldistmap_generated_' + filename+ '.nii.gz')
        #print(file_nii)

        if is_file_nii(file_nii):
            aimsvol = aims.read(file_nii)
            sample = np.asarray(aimsvol)
            subject = re.search('(\d{6})', file_nii).group(1)
            list_sample_id.append(subject)
            list_sample_file.append(sample)

    list_sample_id = np.array(list_sample_id)
    list_sample_file = np.array(list_sample_file)
    # np.save(os.path.join(tgt_dir, 'sub_id_distmap.npy'), list_sample_id)
    # np.save(os.path.join(tgt_dir, 'del_1000_distmap.npy'), list_sample_file)
    np.save(os.path.join(tgt_dir, 'sub_id_skel_all.npy'), list_sample_id)
    np.save(os.path.join(tgt_dir, 'skel_all.npy'), list_sample_file)


if __name__ == '__main__':
    save_to_numpy(
        cropped_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rcrops/',
        tgt_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rcrops')
