"""
Scripts that enables to create a dataframe of numpy arrays from .nii.gz or .nii
images.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from pqdm.processes import pqdm
from p_tqdm import p_map
from soma import aims
# from deep_folding.brainvisa.utils.parallel import define_njobs

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


def save_to_pickle(
        cropped_dir,
        tgt_dir=None,
        file_basename=None,
        parallel=False):
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

    for filename in tqdm(os.listdir(cropped_dir)):
        file_nii = os.path.join(cropped_dir, filename)
        if is_file_nii(file_nii):
            aimsvol = aims.read(file_nii)
            sample = np.asarray(aimsvol)
            subject = re.search('(.*)_cropped_(.*)', file_nii).group(1)
            data_dict[subject] = [sample]

    dataframe = pd.DataFrame.from_dict(data_dict)

    file_pickle_basename = file_basename + '.pkl'
    file_pickle = os.path.join(tgt_dir, file_pickle_basename)
    dataframe.to_pickle(file_pickle)


def save_to_dataframe_format_from_list(
        cropped_dir,
        tgt_dir=None,
        file_basename=None,
        list_sample_id=None,
        list_sample_file=None):
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

    data_dict = {list_sample_id[i]: [list_sample_file[i]]
                 for i in range(len(list_sample_id))}
    dataframe = pd.DataFrame.from_dict(data_dict)

    file_pickle_basename = file_basename + '.pkl'
    file_pickle = os.path.join(tgt_dir, file_pickle_basename)
    dataframe.to_pickle(file_pickle)


def compare_one_array(cropped_dir, list_basename, row):
    index = row[0]
    sub = row[1]
    index_sub = [idx for idx, x in enumerate(list_basename) if str(sub) in x]
    if len(index_sub):
        index_sub = index_sub[0]
    else:
        raise ValueError(f"Subject {sub} not in cropped files")
    subject_file = f"{cropped_dir}/{list_basename[index_sub]}"
    vol = aims.read(subject_file)
    arr_ref = np.asarray(vol)
    return arr_ref


def compare_array_aims_files(subjects, arr, cropped_dir, parallel=True):
    """Compares numpy arrays to subject nifti files"""
    log.info(f"subjects.head() = {subjects.head()}")
    if parallel:
        log.info("Quality check is done in PARALLEL")
        list_nifti = glob.glob(f"{cropped_dir}/*.nii.gz")
        list_basename = [os.path.basename(f) for f in list_nifti]
        log.info(f"list_basename[:3] = {list_basename[:3]}")
        partial_func = partial(compare_one_array, cropped_dir, list_basename)
        enum = [x for x in enumerate(subjects['Subject'])]
        log.info(f"enum subjects[:3] = {enum[:3]}")
        list_arr = p_map(partial_func, enum)
        for index, arr_ref in enumerate(list_arr):
            if not np.array_equal(arr_ref, arr[index, ...]):
                raise ValueError(
                    f"For subject = {list_basename[index]} "
                    f"and index = {index}\n"
                    "arrays do not match")
    else:
        log.info("Quality check is done SERIALLY")
        for index, row in tqdm(subjects.iterrows()):
            sub = row['Subject']
            subject_file = glob.glob(f"{cropped_dir}/{sub}*.nii.gz")[0]
            vol = aims.read(subject_file)
            arr_ref = np.asarray(vol)
            arr_from_array = arr[index, ...]
            if not np.array_equal(arr_ref, arr_from_array):
                raise ValueError(f"For subject = {sub} and index = {index}\n"
                                 "arrays do not match")


def quality_checks(
        csv_file_path,
        npy_array_file_path,
        cropped_dir,
        parallel=False):
    """Checks that the numpy arrays are equal to subject nifti files.

    This is to check that the subjects list in csv file
    match the order set in numpy arrays"""
    arr = np.load(npy_array_file_path, mmap_mode='r')
    subjects = pd.read_csv(csv_file_path, dtype=str)
    compare_array_aims_files(subjects, arr, cropped_dir, parallel)


def get_one_numpy_array(filename, cropped_dir):
    file_nii = os.path.join(cropped_dir, filename)
    if is_file_nii(file_nii):
        aimsvol = aims.read(file_nii)
        sample = np.asarray(aimsvol)
        subject = re.search('(.*)_cropped_(.*)', file_nii).group(1)
        id = os.path.basename(subject)
        if isinstance(sample, np.ndarray):
            return id, sample
        else:
            raise ValueError(
                f"For file={file_nii} and id={id}, "
                "no numpy array has been generated. ")
    else:
        raise ValueError(
            f"file={file_nii} does not look like a nifti file")


def save_to_numpy(
        cropped_dir,
        tgt_dir=None,
        file_basename=None,
        parallel=False):
    """
    Creates a numpy array for each subject.

    Saved these this dataframe to npy format on the target
    directory

    Args:
        cropped_dir: directory containing cropped images
        tgt_dir: directory where to save the numpy array file
        file_basename: final file name = file_basename.npy
    """
    list_sample_id = []
    list_sample_file = []

    log.info("\n\n--------------------------------\n"
             "Now generating numpy array: 4 steps\n"
             "--------------------------------\n")
    log.debug(f"cropped_dir = {cropped_dir}")
    log.info("STEP 1. Now reading cropped dir...")
    listdir = os.listdir(cropped_dir)
    listdir = [filename for filename in listdir
               if is_file_nii(os.path.join(cropped_dir, filename))]
    if parallel:
        log.info("Reading cropped dir is done in PARALLEL")
        partial_func = partial(get_one_numpy_array, cropped_dir=cropped_dir)
        list_result = p_map(partial_func, sorted(listdir))
        list_sample_id, list_sample_file =\
            [x for x, y in list_result], [y for x, y in list_result]
    else:
        log.info("Reading cropped dir is done SERIALLY")
        for filename in tqdm(sorted(listdir)):
            file_nii = os.path.join(cropped_dir, filename)
            if is_file_nii(file_nii):
                aimsvol = aims.read(file_nii)
                sample = np.asarray(aimsvol)
                subject = re.search('(.*)_cropped_(.*)', file_nii).group(1)
                list_sample_id.append(os.path.basename(subject))
                list_sample_file.append(sample)

    log.info("STEP 2. Now writing subject name file...")
    # Writes subject ID csv file
    subject_df = pd.DataFrame(list_sample_id, columns=["Subject"])
    subject_df.to_csv(os.path.join(tgt_dir, file_basename + '_subject.csv'),
                      index=False)
    np.save(os.path.join(tgt_dir, 'sub_id.npy'), list_sample_id)

    log.info("STEP 3. Now saving to numpy array...")
    # Writes volumes as numpy arrays
    list_sample_file = np.array(list_sample_file)
    np.save(os.path.join(tgt_dir, file_basename + '.npy'), list_sample_file)

    # Quality_checks
    log.info("STEP 4. Now performing checks on numpy arrays...")
    quality_checks(
        os.path.join(tgt_dir, file_basename + '_subject.csv'),
        os.path.join(tgt_dir, file_basename + '.npy'),
        cropped_dir,
        parallel=parallel)

    return list_sample_id, list_sample_file


if __name__ == '__main__':
    # save_to_pickle(
    #     cropped_dir='/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/Rlabels/',
    #     tgt_dir='/neurospin/dico/data/deep_folding/current/crops/SC/mask/sulcus_based/2mm/',
    #     file_basename='Rlabels')
    save_to_numpy(
        cropped_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rcrops/',
        tgt_dir='/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/1mm/SC/no_mask/Rcrops',
        file_basename='Rlabels')
