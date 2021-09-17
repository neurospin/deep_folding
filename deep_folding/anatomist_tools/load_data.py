"""
Scripts that enables to create a dataframe of numpy arrays from .nii.gz or .nii
images.
"""
from __future__ import division
import os

from soma import aims

import pandas as pd
import numpy as np
import re


def is_file_nii(filename):
    """Tests if file is nii file

    Args:
        filename: string giving file name with full path

    Returns:
        is_file_nii: boolean stating if file is nii file
    """
    is_file_nii = os.path.isfile(filename)\
                  and '.nii' in filename \
                  and '.minf' not in filename \
                  and 'normalized' in filename
    return is_file_nii


def fetch_data(cropped_dir, tgt_dir=None, side=None):
    """
    Creates a dataframe of data with a column for each subject and associated
    np.array. Generation a dataframe of "normal" images and a dataframe of
    "abnormal" images. Saved these this dataframe to pkl format on the target
    directory

    Args:
        cropped_dir: directory containing cropped images
        tgt_dir: directory where to save the pickle file
        side: hemisphere side, either 'L' for left or 'R' for right hemisphere
    """

    data_dict = dict()

    for filename in os.listdir(cropped_dir):
        file_nii = os.path.join(cropped_dir, filename)
        if is_file_nii(file_nii):
            aimsvol = aims.read(file_nii)
            sample = np.asarray(aimsvol)
            subject = re.search('(\d{4,12})', file_nii).group(1)
            data_dict[subject] = [sample]

    dataframe = pd.DataFrame.from_dict(data_dict)

    file_pickle_basename = side + 'skeleton.pkl'
    file_pickle = os.path.join(tgt_dir, file_pickle_basename)
    dataframe.to_pickle(file_pickle)


if __name__ == '__main__':
    fetch_data(cropped_dir='/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm/Rcrops',
               tgt_dir='/neurospin/dico/data/deep_folding/data/crops/SC/sulcus_based/2mm',
               side='R')
