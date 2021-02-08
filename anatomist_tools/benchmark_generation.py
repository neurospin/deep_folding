# The aim of this script is to generate a benchmark of sulcal abnormalities.
# In this script, abnormalities are defined as skeleton with one simple surface
# missing. This simple surface must be completely in the bounding box of interest
#(currently S.T.s.ter.asc.ant/post Left) and include a minimum number of voxels
# (in order that the anomaly is big enough to be considered as abnormal).

# Modules import
from soma import aims
import numpy as np
from glob import glob
import random
import pandas as pd
import os

def generate(b_num, side, region, ss_size):
    """
    IN: b_num: benchmark number
        side: hemisphere, str, whether 'L' or 'R',
        region: region of the brain
        ss_size: Minimal size of simple surface to suppress
    OUT: altered skeletons generated
         list of subjects altered or original
    """
    # List of right handed subjects
    right_handed = pd.read_csv('/neurospin/dico/lguillon/hcp_info/right_handed.csv')
    subjects_list = list(right_handed['Subject'])

    data_dir = '/neurospin/hcp/ANALYSIS/3T_morphologist/' # folder containing all
                                                          # HCP subjects folder
    saving_dir = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark' + str(b_num) + '/0_' + side + 'side/'
    #folder_list = glob(data_dir + '*') # get list of all subjects folder
    random.shuffle(subjects_list)

    # Saving of simple surfaces satisfying both criteria: completely inside the
    # bounding box defined for the crop and including at least 500 voxels

    abnormality_test = []
    if region == 'S.T.s':
        bbmin = [34.05417004,  39.15836927, -65.53190718] # Obtained with bbox_definition.py script
        bbmax = [71.73703268, 85.71765662, -2.74244928] # Obtained with bbox_definition.py script

    for sub in subjects_list:
        print(sub)
        if os.path.isdir(data_dir + str(sub)):
            surfaces = dict()

            graph = aims.read(data_dir + str(sub) + '/t1mri/default_acquisition/default_analysis/folds/3.1/default_session_auto/'+ side + str(sub) + '_default_session_auto.arg')
            skel = aims.read(data_dir + str(sub) + '/t1mri/default_acquisition/default_analysis/segmentation/' + side + 'skeleton_' + str(sub) + '.nii.gz')

            for v in graph.vertices():
                if 'label' in v:
                    bbmin_surface = v['Tal_boundingbox_min']
                    bbmax_surface = v['Tal_boundingbox_max']
                    bck_map = v['aims_ss']

                    if all([a >= b for (a, b) in zip(bbmin_surface, bbmin)]) and all([a <= b for (a, b) in zip(bbmax_surface, bbmax)]):
                        for bucket in bck_map:
                            if bucket.size() > ss_size: # In order to keep only large enough simple surfaces
                                surfaces[len(surfaces)] = v

            print(len(surfaces.keys()))
            if len(surfaces.keys()) > 0:
                # Suppression of one random simple surface (satisfying both criteria)
                surface = random.randint(0, len(surfaces)-1)
                print(surfaces[surface]['label'])

                bck_map = surfaces[surface]['aims_ss']
                for voxel in bck_map[0].keys():
                    skel.setValue(0, voxel[0], voxel[1], voxel[2])

                bck_map_bottom = surfaces[surface]['aims_bottom']
                for voxel in bck_map_bottom[0].keys():
                    skel.setValue(0, voxel[0], voxel[1], voxel[2])

                # Writting of output graph
                fileout = saving_dir + 'output_skeleton_' + str(sub) + '.nii.gz'
                print('writing altered skeleton to', fileout)
                aims.write(skel, fileout)

                # Addition of modified graph to abnormality_test set
                abnormality_test.append(sub)
                if len(abnormality_test) == 70:
                    break

    # Train, validation and test separation
    # It is important that examples of this benchmark aren't used during training and
    # validation phase. In order to make sure this is the case, 2 lists are created.
    nor_set = list(set(subjects_list) - set(abnormality_test))

    print('Dataset split, train + val + normal test: ', len(nor_set),
          'abnormal test: ', len(abnormality_test))

    df_train = pd.DataFrame(nor_set)
    df_train.to_csv(saving_dir + 'train.csv')

    df_abnor_test = pd.DataFrame(abnormality_test)
    df_abnor_test.to_csv(saving_dir + 'abnormality_test.csv')


if __name__ == '__main__':
    benchmark_num = 2
    side = 'L'
    region = 'S.T.s'
    ss_size = 600
    generate(benchmark_num, side, region, ss_size)
