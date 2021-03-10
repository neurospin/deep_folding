"""
Scripts that enables to create a dataframe of numpy arrays from .nii.gz or .nii
images.
"""
from __future__ import division
import os

import anatomist.api as anatomist
from soma import aims

import pandas as pd
import numpy as np
import re


def fetch_data(root_dir, save_dir=None, side=None):
    """
    Creates a dataframe of data with a column for each subject and associated
    np.array. Generation a dataframe of "normal" images and a dataframe of
    "abnormal" images. Saved these two dataframes to pkl format on
    Neurospin/dico/lguillon/data directory.
    -----------
    Parameter:
    root_dir: directory of training images
    """

    data = ['train', 'test']
    for phase in data:
        phase=''
        data_dict = dict()
        for filename in os.listdir(root_dir+phase):
            file = os.path.join(root_dir+phase, filename)
            #print(filename)
            if os.path.isfile(file) and '.nii' in file and '.minf' not in file and 'normalized' in file:
		aimsvol = aims.read(file)
                sample = np.asarray(aimsvol).T
                if input == 'skeleton':
                    filename = re.search('_(\d{6})', file).group(1)
                else:
                    # filename = re.search('-(\d{6})', file).group(1)
                    #filename = re.search('_(\d{6})', file).group(1)
                    #filename = re.search('(\d{12})', file).group(1)
                    filename = re.search('(\d{6})', file).group(1)

                data_dict[filename] = [sample]
                #print(filename)

        dataframe = pd.DataFrame.from_dict(data_dict)

        if save_dir:
            dataframe.to_pickle(save_dir + side + 'skeleton.pkl')
        else:
            dataframe.to_pickle('/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark2/abnormal_skeleton_left.pkl')


if __name__ == '__main__':
    input = 'skeleton'
    # input = 'raw'

    if input == 'raw':
        # Raw MRIs - crop
        fetch_data('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/right_hemi/normalized_crops/')

    else:
        # Skeletons
        fetch_data('/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark2/0_Lside/')
