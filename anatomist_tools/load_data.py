from __future__ import division
import os

import anatomist.api as anatomist
from soma import aims
a = anatomist.Anatomist()

#from pynet_transforms import *
import pandas as pd
import numpy as np
import re


def fetch_data(root_dir):
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
        #skeleton_array = []
        data_dict = dict()
        for filename in os.listdir(root_dir+phase):
            file = os.path.join(root_dir+phase, filename)
            print(filename)
            if os.path.isfile(file) and '.nii' in file and '.minf' not in file and 'normalized' in file:
                vol = a.loadObject(file)
                aimsvol = a.toAimsObject(vol)
                sample = np.asarray(aimsvol.volume()).T
                if input == 'skeleton':
                    filename = re.search('_(\d{6})', file).group(1)
                else:
                    # filename = re.search('-(\d{6})', file).group(1)
                    #filename = re.search('_(\d{6})', file).group(1)
                    filename = re.search('(\d{12})', file).group(1)

                data_dict[filename] = [sample]
                print(filename)
                #skeleton_array.append(sample)

        dataframe = pd.DataFrame.from_dict(data_dict)

        # skeleton
        # dataframe.to_pickle('/neurospin/dico/lguillon/data/data_skeleton_saved_1mm_%s.pkl' % phase)
        # dataframe.to_pickle('/neurospin/dico/lguillon/data/sk_stripped_crop_2_%s.pkl' % phase)
        # dataframe.to_pickle('/neurospin/dico/lguillon/data/crop_norm_spm_%s.pkl' % phase)
        # dataframe.to_pickle('/neurospin/dico/lguillon/data/crop_norm_spm_skel_%s.pkl' % phase)
        # dataframe.to_pickle('/neurospin/dico/lguillon/hcp_cs_crop/exp_crop_size/shift_minus5_xy/shift_minus5_xy_%s.pkl' % phase)
        # dataframe.to_pickle('/neurospin/dico/lguillon/hcp_cs_crop/sts_crop/sts_crop.pkl')
        # dataframe.to_pickle('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/sts_crop_skeleton_right.pkl')
        dataframe.to_pickle('/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark2/abnormal_skeleton_left.pkl')
        #np.save(root_dir+'_skeleton_saved_%s' % phase, skeleton_array)

if __name__ == '__main__':
    input = 'skeleton'
    # input = 'raw'

    if input == 'raw':
        # Raw MRIs - crop
        # fetch_data('/home/lg261972/manip/hcp_cs/hcp_cs_sep/data/')
        # Crop with same procedure as skeletons
        #fetch_data('/neurospin/dico/lguillon/hcp_cs_crop/sk_stripped_')
        # fetch_data('/neurospin/dico/lguillon/hcp_cs_crop/v2/')
        # fetch_data('/neurospin/dico/lguillon/hcp_cs_crop/v4_norm_spm/')
        #fetch_data('/neurospin/dico/lguillon/hcp_cs_crop/exp_crop_size/shift_minus5_xy/')
        # fetch_data('/neurospin/dico/lguillon/hcp_cs_crop/sts_crop/')
        # fetch_data('/neurospin/dico/lguillon/skeleton/sts_crop/right_hemi/normalized_crop/')
        fetch_data('/neurospin/dico/lguillon/aims_detection/aims_crop/skeleton/right_hemi/normalized_crops/')

    else:
        # Skeletons
        #fetch_data('/neurospin/dico/lguillon/skeleton/skeleton_crop_')
        #fetch_data('/neurospin/dico/lguillon/skeleton/normalized_skeleton/')
        fetch_data('/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark2/0_Lside/')
