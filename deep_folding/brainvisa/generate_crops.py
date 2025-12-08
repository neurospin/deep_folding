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

"""Creating npy file from T1 MRI datas
The aim of this script is to create dataset of cropped skeletons from MRIs
saved in a .npy file.
We read resampled skeleton files
Several steps are required: crop and .npy generation
  Typical usage
  -------------
  You can use this program by first entering in the brainvisa environment
  (here brainvisa 5.0.0 installed with singurity) and launching the script
  from the terminal:
  >>> bv bash
  >>> python3 generate_crops.py
  Alternatively, you can launch the script in the interactive terminal ipython:
  >>> %run generate_crops.py
"""

import argparse
import glob
import os
import re
import sys
from os.path import join
from os.path import basename

import numpy as np
import pandas as pd
import scipy.ndimage

from deep_folding.brainvisa import exception_handler
from deep_folding.brainvisa.utils.save_data import save_to_numpy
from deep_folding.brainvisa.utils.bbox import compute_max_box
from deep_folding.brainvisa.utils.folder import create_folder
from deep_folding.brainvisa.utils.logs import LogJson
from deep_folding.brainvisa.utils.logs import setup_log
from deep_folding.brainvisa.utils.parallel import define_njobs
from deep_folding.brainvisa.utils.mask import compute_centered_mask
from deep_folding.brainvisa.utils.mask import compute_simple_mask
from deep_folding.brainvisa.utils.mask import compute_intersection_mask
from deep_folding.brainvisa.utils.subjects import get_number_subjects
from deep_folding.brainvisa.utils.subjects import select_subjects_int
from deep_folding.brainvisa.utils.quality_checks import \
    compare_number_aims_files_with_expected, \
    get_not_processed_cropped_files, \
    compare_number_aims_files_with_number_in_source, \
    save_list_to_csv
from deep_folding.brainvisa.utils.sulcus import complete_sulci_name
from deep_folding.config.logs import set_file_logger
from p_tqdm import p_map
from soma import aims

# Import constants
from deep_folding.brainvisa.utils.constants import \
    _ALL_SUBJECTS, _RESAMPLED_SKELETON_DIR_DEFAULT, \
    _BBOX_DIR_DEFAULT, _MASK_DIR_DEFAULT, \
    _CROP_DIR_DEFAULT, \
    _SIDE_DEFAULT, _CROPPING_TYPE_DEFAULT, \
    _COMBINE_TYPE_DEFAULT, _INPUT_TYPE_DEFAULT, \
    _SULCUS_DEFAULT, _NO_MASK_DEFAULT, \
    _DILATION_DEFAULT, _THRESHOLD_DEFAULT, \
    _NB_JOBS_DEFAULT

# Defines logger
log = set_file_logger(__file__)


def crop_bbox(file_src: str, file_cropped: str,
              bbmin: np.array, bbmax: np.array):
    """Crops according to bounding box"""

    # Read source hemisphere file
    vol = aims.read(file_src)

    # Crops volume according to bounding box
    vol_cropped = aims.VolumeView(vol, bbmin, bbmax - bbmin)
    aims.write(vol_cropped, file_cropped)


def filter_mask(mask: aims.Volume):
    """Smooths the mask with Gaussian Filter
    """
    arr = np.asarray(mask)
    arr_filter = scipy.ndimage.gaussian_filter(
        arr.astype(float),
        sigma=0.5,
        order=0,
        output=None,
        mode='reflect',
        truncate=4.0)
    arr[:] = (arr_filter > 0.001).astype(int)


def crop_mask(file_src, file_cropped, mask, bbmin, bbmax, side,
              no_mask=_NO_MASK_DEFAULT):
    """Crops according to mask"""
    vol = aims.read(file_src)

    arr = np.asarray(vol)

    arr_mask = np.asarray(mask)
    if no_mask:
        pass
    else:
        arr[arr_mask == 0] = 0

    # bbmin = np.array([14, 89, 24])
    # bbmax = np.array([92, 152, 112])
    # bbmin = np.array([24, 64, 28])
    # bbmax = np.array([102, 127, 116])
    # For benchmark asymmetry
    # bbmin = np.array([92, 84, 24])
    # bbmax = np.array([170, 147, 112])

    log.debug(f"bbmin = {bbmin.tolist()}")
    log.debug(f"bbmax = {bbmax.tolist()}")
    log.debug(f"size = {(bbmax-bbmin).tolist()}")
    # Crops volume according to mask bounding box
    vol_cropped = aims.VolumeView(vol, bbmin, bbmax - bbmin)
    aims.write(vol_cropped, file_cropped)
    # # Crops mask according to mask bounding box
    # file_mask = os.path.dirname(os.path.dirname(file_cropped))
    # mask_cropped = aims.VolumeView(mask, bbmin, bbmax - bbmin)
    # aims.write(mask_cropped,
    #            f"{file_mask}/{side}mask_cropped.nii.gz")


def quality_checks(crop_dir, side):
    s = np.load(f"{crop_dir}/{side}skeleton.npy")
    f = np.load(f"{crop_dir}/{side}label.npy")

    # checks if same voxel position
    assert (s.shape == f.shape), (
        f"Skeleton and foldlabel of different shapes: {s.shape} != {f.shape}")
    assert (f[s == 0].sum() == 0), (
        f"Foldlabel and skeleton arrays with different non-zero positions: "
        f"{(f[s == 0] != 0).sum()} different non-zero positions")
    assert (s[f == 0].sum() == 0), (
        f"Foldlabel and skeleton arrays with different non-zero positions: "
        f"{(s[f == 0] != 0).sum()} different non-zero positions")

    # Checks if subjects are equal between foldlabel and skeleton
    dff = pd.read_csv(f"{crop_dir}/{side}label_subject.csv")
    dfs = pd.read_csv(f"{crop_dir}/{side}skeleton_subject.csv")
    assert (dff.equals(dfs)), \
        "List of subjects for foldlabel and skeleton are not equal"

    # Checks if numpy arrays and csvs are consistent
    assert (s.shape[0] == len(dfs)), \
        "Number of skeleton subjects differs between numpy array and csv"
    assert (f.shape[0] == len(dff)), \
        "Number of foldlabel subjects differs between numpy array and csv"


def quality_checks_extremities(crop_dir, side):
    s = np.load(f"{crop_dir}/{side}skeleton.npy")
    f = np.load(f"{crop_dir}/{side}extremities.npy")

    # checks if same voxel position
    assert (s.shape == f.shape), (
        "Skeleton and extremities of different shapes: "
        f"{s.shape} != {f.shape}")

    # Checks if subjects are equal between foldlabel and skeleton
    dff = pd.read_csv(f"{crop_dir}/{side}extremities_subject.csv")
    dfs = pd.read_csv(f"{crop_dir}/{side}skeleton_subject.csv")
    assert (dff.equals(dfs)), \
        "List of subjects for extremities and skeleton are not equal"

    # Checks if numpy arrays and csvs are consistent
    assert (s.shape[0] == len(dfs)), \
        "Number of skeleton subjects differs between numpy array and csv"
    assert (f.shape[0] == len(dff)), \
        "Number of extremities subjects differs between numpy array and csv"


class CropGenerator:
    """Generates cropped skeleton files and corresponding npy file
    """

    def __init__(self,
                 src_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
                 crop_dir=_CROP_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 dilation=_DILATION_DEFAULT,
                 threshold=_THRESHOLD_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 cropping_type=_CROPPING_TYPE_DEFAULT,
                 combine_type=_COMBINE_TYPE_DEFAULT,
                 parallel=False,
                 no_mask=_NO_MASK_DEFAULT,
                 njobs=_NB_JOBS_DEFAULT):
        """Inits with list of directories and list of sulci
        Args:
            src_dir: folder containing generated skeletons, labels or distmaps
            crop_dir: name of output directory for crops with full path
            bbox_dir: directory containing bbox json files
                    (generated using compute_bounding_box.py)
            mask_dir: directory containing mask files
                    (generated using compute_mask.py)
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left,
                                   or R for right hemisphere)
            cropping_type: cropping type, either mask, or bbox
            combine_type: if True, combines sulci (in this case, order matters)
        """

        self.crop_dir = crop_dir
        self.side = side
        # Transforms sulcus in a list of sulci
        self.list_sulci = ([list_sulci] if isinstance(list_sulci, str)
                           else list_sulci)
        self.list_sulci = complete_sulci_name(self.list_sulci, self.side)
        self.bbox_dir = bbox_dir
        self.mask_dir = mask_dir
        self.dilation = dilation
        self.threshold = threshold
        self.cropping_type = cropping_type
        self.combine_type = combine_type
        self.parallel = parallel
        self.no_mask = no_mask
        self.njobs = njobs
        print(self.no_mask)

        # Names of files in function of dictionary:
        #               keys -> 'subject' and 'side'
        # Generated skeleton from folding graphs
        self.src_dir = join(src_dir, self.side)

        # Initialization of bounding box coordinates
        self.bbmin = np.zeros(3)
        self.bbmax = np.zeros(3)

    def crop_one_file(self, subject_id):
        """Crops one file
        Args:
            subject_id: string giving the subject ID
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        # subject = {'subject': subject_id, 'side': self.side}
        # FOR TISSIER
        # subject_id = re.search('([ae\d]{5,6})', subject_id).group(0)

        # Skeleton file name
        file_src = self.src_file % {
            'subject': subject_id, 'side': self.side}

        if os.path.exists(file_src):
            # Creates output (cropped) file name
            file_cropped = join(
                self.cropped_samples_dir,
                self.cropped_file % {
                    'subject': subject_id,
                    'side': self.side})

            # Cropping of skeleton image
            if self.cropping_type == 'bbox':
                crop_bbox(file_src, file_cropped,
                          self.bbmin, self.bbmax)
            else:
                crop_mask(file_src, file_cropped,
                          self.mask, self.bbmin, self.bbmax, self.side,
                          self.no_mask)
        else:
            raise FileNotFoundError(f"{file_src} not found")

    def crop_files(self, nb_subjects=_ALL_SUBJECTS):
        """Crop nii files
        The programm loops over all subjects from the input (source) directory.
        Args:
            nb_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        if nb_subjects:

            if os.path.isdir(self.src_dir):
                files = glob.glob(f"{self.src_dir}/*.nii.gz")
                log.debug(f"Nifti files in {self.src_dir} = {files}")
                log.debug(f"Regular expresson is: {self.expr}")

                # Creates target directories
                create_folder(self.crop_dir)
                create_folder(self.cropped_samples_dir)

                # Generates list of subjects not treated yet
                not_processed_files = get_not_processed_cropped_files(
                    self.src_dir,
                    self.cropped_samples_dir)

                if len(files):
                    list_not_processed_subjects = [
                        re.search(self.expr, basename(dI))[1]
                        for dI in not_processed_files]
                    list_all_subjects = [
                        re.search(self.expr, basename(dI))[1]
                        for dI in files]
                else:
                    raise ValueError(f"no nifti files in {self.src_dir}")
            else:
                raise NotADirectoryError(
                    f"{self.src_dir} doesn't exist or is not a directory")

            if len(list_all_subjects):
                # Gives the possibility to list
                # only the first nb_subjects
                list_subjects = select_subjects_int(
                                    list_all_subjects,
                                    list_not_processed_subjects,
                                    nb_subjects)

                log.info(f"Expected number of subjects = {len(list_subjects)}")
                log.info(f"list_subjects[:5] = {list_subjects[:5]}")
                log.debug(f"list_subjects = {list_subjects}")

                # Creates target and cropped directory
                create_folder(self.crop_dir)
                create_folder(self.cropped_samples_dir)

                # Crops mask according to mask bounding box, for debugging
                mask_cropped = aims.VolumeView(self.mask,
                                               self.bbmin,
                                               self.bbmax - self.bbmin)
                aims.write(mask_cropped,
                           f"{self.crop_dir}/{self.side}mask_cropped.nii.gz")

                # Writes number of subjects and directory names to json file
                dict_to_add = {
                    'nb_subjects': len(list_subjects),
                    'src_dir': self.src_dir,
                    'bbox_dir': self.bbox_dir,
                    'mask_dir': self.mask_dir,
                    'side': self.side,
                    'list_sulci': self.list_sulci,
                    'bbmin': self.bbmin.tolist(),
                    'bbmax': self.bbmax.tolist(),
                    'size': (
                        self.bbmax - self.bbmin).tolist(),
                    'crop_dir': self.crop_dir,
                    'cropped_skeleton_dir': self.cropped_samples_dir,
                    'cropping_type': self.cropping_type,
                    'combine_type': self.combine_type,
                    'no_mask': self.no_mask}
                self.json.update(dict_to_add=dict_to_add)

                if self.parallel:
                    log.info(
                        "PARALLEL MODE: subjects are in parallel")
                    p_map(
                        self.crop_one_file,
                        list_subjects,
                        num_cpus=define_njobs())
                else:
                    log.info(
                        "SERIAL MODE: subjects are scanned serially")
                    for sub in list_subjects:
                        self.crop_one_file(sub)
            else:
                list_subjects = []
                log.info(
                    "There is no subject or there is no subject to process"
                    "in the source directory")

            # Checks if there is expected number of generated files
            compare_number_aims_files_with_expected(self.cropped_samples_dir,
                                                    list_subjects)

            # Checks if number of generated files == number of src files
            crop_files, src_files = \
                compare_number_aims_files_with_number_in_source(
                    self.cropped_samples_dir,
                    self.src_dir)
            not_processed_files = get_not_processed_cropped_files(
                self.src_dir, self.cropped_samples_dir)
            save_list_to_csv(not_processed_files,
                             f"{self.crop_dir}/not_processed_files.csv")

    def compute_bounding_box_or_mask(self, nb_subjects):
        """Computes bounding box or mask
        Args:
            nb_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1)."""

        if nb_subjects:
            if self.cropping_type == 'bbox':
                self.bbmin, self.bbmax = \
                    compute_max_box(sulci_list=self.list_sulci,
                                    side=self.side,
                                    talairach_box=False,
                                    src_dir=self.bbox_dir)
            elif self.cropping_type == 'mask':
                if self.combine_type:
                    # /!\ SPECIFIC FOR THE CINGULATE REGION STUDY
                    # (2022, CHAVAS, GAUDIN & CHAVAS, GUILLON)
                    self.mask, self.bbmin, self.bbmax = \
                        compute_centered_mask(sulci_list=self.list_sulci,
                                              side=self.side,
                                              mask_dir=self.mask_dir)
                else:
                    self.mask, self.bbmin, self.bbmax = \
                        compute_simple_mask(sulci_list=self.list_sulci,
                                            side=self.side,
                                            mask_dir=self.mask_dir,
                                            dilation=self.dilation,
                                            threshold=self.threshold)
                mask_filename = \
                    f"{self.crop_dir}/{self.side}mask_{self.input_type}.nii.gz"
                aims.write(
                    self.mask,
                    mask_filename)
            elif self.cropping_type == 'mask_intersect':
                self.mask, self.bbmin, self.bbmax = \
                    compute_intersection_mask(sulci_list=self.list_sulci,
                                              side=self.side,
                                              mask_dir=self.mask_dir,
                                              dilation=self.dilation,
                                              threshold=self.threshold)
                mask_filename = \
                    f"{self.crop_dir}/{self.side}mask_{self.input_type}.nii.gz"
                aims.write(
                    self.mask,
                    mask_filename)
            else:
                raise ValueError(
                    "cropping_type must be either "
                    "\'bbox\' or \'mask\' or \'mask_intersect\'")

    def compute(self, nb_subjects=_ALL_SUBJECTS):
        """Main API to create numpy files
        The programm loops over all subjects from the input (source) directory.
        Args:
            nb_subjects: integer giving the number of subjects to analyze,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        self.json.write_general_info()

        # Computes bounding box or mask
        self.compute_bounding_box_or_mask(nb_subjects=nb_subjects)

        # Generate cropped files
        self.crop_files(nb_subjects=nb_subjects)

        # Creation of .npy file containing all subjects
        if nb_subjects:
            list_sample_id, list_sample_file = \
                save_to_numpy(cropped_dir=self.cropped_samples_dir,
                              tgt_dir=self.crop_dir,
                              file_basename=self.file_basename_npy,
                              parallel=self.parallel)


class SkeletonCropGenerator(CropGenerator):
    """Generates cropped skeleton files and corresponding npy file
    """

    def __init__(self,
                 src_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
                 crop_dir=_CROP_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 cropping_type=_CROPPING_TYPE_DEFAULT,
                 combine_type=_COMBINE_TYPE_DEFAULT,
                 parallel=False,
                 no_mask=_NO_MASK_DEFAULT,
                 threshold=_THRESHOLD_DEFAULT,
                 dilation=_DILATION_DEFAULT,
                 njobs=_NB_JOBS_DEFAULT):
        """Inits with list of directories and list of sulci
        Args:
            src_dir: folder containing generated skeletons or labels
            crop_dir: name of output directory for crops with full path
            bbox_dir: directory containing bbox json files
                    (generated using compute_bounding_box.py)
            mask_dir: directory containing mask files
                    (generated using compute_mask.py)
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left,
                                   or R for right hemisphere)
            cropping_type: cropping type, either mask, bbox, or mask_intersect
            combine_type: if True, combines sulci (in this case, order matters)
            parallel: if True, parallel computation
        """
        super(SkeletonCropGenerator, self).__init__(
            src_dir=src_dir, crop_dir=crop_dir,
            bbox_dir=bbox_dir, mask_dir=mask_dir,
            list_sulci=list_sulci, side=side,
            cropping_type=cropping_type, combine_type=combine_type,
            parallel=parallel, no_mask=no_mask,
            threshold=threshold, dilation=dilation
        )

        # Directory where to store cropped skeleton files
        self.cropped_samples_dir = join(self.crop_dir, self.side + 'crops')

        # Names of files in function of dict: keys -> 'subject' and 'side'
        # Generated skeleton from folding graphs
        self.src_file = join(
            self.src_dir,
            '%(side)sresampled_skeleton_%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.cropped_file = '%(subject)s_cropped_skeleton.nii.gz'

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.resampled_skeleton_(.*).nii.gz$'

        # Creates json log class
        json_file = join(self.crop_dir, self.side + 'skeleton.json')
        self.json = LogJson(json_file)

        # Creates npys file name
        self.file_basename_npy = self.side + 'skeleton'
        self.file_basename_pickle = self.side + 'skeleton'

        self.input_type = 'skeleton'


class FoldLabelCropGenerator(CropGenerator):
    """Generates cropped skeleton files and corresponding npy file
    """

    def __init__(self,
                 src_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
                 crop_dir=_CROP_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 cropping_type=_CROPPING_TYPE_DEFAULT,
                 combine_type=_COMBINE_TYPE_DEFAULT,
                 parallel=False,
                 no_mask=_NO_MASK_DEFAULT,
                 threshold=_THRESHOLD_DEFAULT,
                 dilation=_DILATION_DEFAULT,
                 njobs=_NB_JOBS_DEFAULT):
        """Inits with list of directories and list of sulci
        Args:
            src_dir: folder containing generated labels
            crop_dir: name of output directory for crops with full path
            bbox_dir: directory containing bbox json files
                    (generated using compute_bounding_box.py)
            mask_dir: directory containing mask files
                    (generated using compute_mask.py)
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left,
                                   or R for right hemisphere)
            cropping_type: cropping type, either mask, or bbox,
                                   or mask_intersect
            combine_type: if True, combines sulci (in this case, order matters)
            parallel: if True, parallel computation
        """
        super(FoldLabelCropGenerator, self).__init__(
            src_dir=src_dir, crop_dir=crop_dir,
            bbox_dir=bbox_dir, mask_dir=mask_dir,
            list_sulci=list_sulci, side=side,
            cropping_type=cropping_type, combine_type=combine_type,
            parallel=parallel, no_mask=no_mask,
            threshold=threshold, dilation=dilation,
        )

        # Directory where to store cropped skeleton files
        self.cropped_samples_dir = join(self.crop_dir, self.side + 'labels')

        # Names of files in function of dictionary: keys -> 'subject'+'side'
        # Generated skeleton from folding graphs
        self.src_file = join(
            self.src_dir,
            '%(side)sresampled_foldlabel_%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.cropped_file = '%(subject)s_cropped_foldlabel.nii.gz'

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.resampled_foldlabel_(.*).nii.gz$'

        # Creates json log class
        json_file = join(self.crop_dir, self.side + 'foldlabel.json')
        self.json = LogJson(json_file)

        # Creates npys file name
        self.file_basename_npy = self.side + 'label'
        self.file_basename_pickle = self.side + 'label'

        self.input_type = 'foldlabel'


class ExtremitiesCropGenerator(CropGenerator):
    """Generates cropped skeleton files and corresponding npy file
    """

    def __init__(self,
                 src_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
                 crop_dir=_CROP_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 cropping_type=_CROPPING_TYPE_DEFAULT,
                 combine_type=_COMBINE_TYPE_DEFAULT,
                 parallel=False,
                 no_mask=_NO_MASK_DEFAULT,
                 threshold=_THRESHOLD_DEFAULT,
                 dilation=_DILATION_DEFAULT,
                 njobs=_NB_JOBS_DEFAULT):
        """Inits with list of directories and list of sulci
        Args:
            src_dir: folder containing generated extremities
            crop_dir: name of output directory for crops with full path
            bbox_dir: directory containing bbox json files
                    (generated using compute_bounding_box.py)
            mask_dir: directory containing mask files
                    (generated using compute_mask.py)
            list_sulci: list of sulcus names
            side: hemisphere side (either L for left,
                                   or R for right hemisphere)
            cropping_type: cropping type, either mask, or bbox,
                                   or mask_intersect
            combine_type: if True, combines sulci (in this case, order matters)
            parallel: if True, parallel computation
        """
        super(ExtremitiesCropGenerator, self).__init__(
            src_dir=src_dir, crop_dir=crop_dir,
            bbox_dir=bbox_dir, mask_dir=mask_dir,
            list_sulci=list_sulci, side=side,
            cropping_type=cropping_type, combine_type=combine_type,
            parallel=parallel, no_mask=no_mask,
            threshold=threshold, dilation=dilation
        )

        # Directory where to store cropped skeleton files
        self.cropped_samples_dir = join(self.crop_dir,
                                        self.side + 'extremities')

        # Names of files in function of dictionary: keys -> 'subject'+'side'
        # Generated skeleton from folding graphs
        self.src_file = join(
            self.src_dir,
            '%(side)sresampled_extremities_%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.cropped_file = '%(subject)s_cropped_extremities.nii.gz'

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.resampled_extremities_(.*).nii.gz$'

        # Creates json log class
        json_file = join(self.crop_dir, self.side + 'extremities.json')
        self.json = LogJson(json_file)

        # Creates npys file name
        self.file_basename_npy = self.side + 'extremities'
        self.file_basename_pickle = self.side + 'extremities'

        self.input_type = 'extremities'


class DistMapCropGenerator(CropGenerator):
    """Generates cropped skeleton files and corresponding npy file
    """

    def __init__(self,
                 src_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
                 crop_dir=_CROP_DIR_DEFAULT,
                 bbox_dir=_BBOX_DIR_DEFAULT,
                 mask_dir=_MASK_DIR_DEFAULT,
                 list_sulci=_SULCUS_DEFAULT,
                 side=_SIDE_DEFAULT,
                 cropping_type=_CROPPING_TYPE_DEFAULT,
                 combine_type=_COMBINE_TYPE_DEFAULT,
                 parallel=False,
                 no_mask=_NO_MASK_DEFAULT,
                 threshold=_THRESHOLD_DEFAULT,
                 dilation=_DILATION_DEFAULT):
        """Inits with list of directories and list of sulci
        Args:
            src_dir: folder containing generated skeletons, labels or distmaps
            crop_dir: name of output directory for crops with full path
            bbox_dir: directory containing bbox json files
                    (generated using compute_bounding_box.py)
            mask_dir: directory containing mask files
                    (generated using compute_mask.py)
            list_sulci: list of sulcus names
            side: hemisphere side (L for left, or R for right hemisphere)
            cropping_type: cropping type, either mask, bbox, or mask_intersect
            combine_type: if True, combines sulci (in this case, order matters)
            parallel: if True, parallel computation
        """
        super(DistMapCropGenerator, self).__init__(
            src_dir=src_dir, crop_dir=crop_dir,
            bbox_dir=bbox_dir, mask_dir=mask_dir,
            list_sulci=list_sulci, side=side,
            cropping_type=cropping_type, combine_type=combine_type,
            parallel=parallel, no_mask=no_mask,
            threshold=threshold, dilation=dilation
        )

        # Directory where to store cropped skeleton files
        self.cropped_samples_dir = join(self.crop_dir, self.side + 'distmaps')

        # Names of files in function of dictionary: keys = 'subject' and 'side'
        # Generated skeleton from folding graphs
        self.src_file = join(
            self.src_dir,
            '%(side)sresampled_distmap_%(subject)s.nii.gz')

        # Names of files in function of dictionary: keys -> 'subject' and
        # 'side'
        self.cropped_file = '%(subject)s_cropped_distmap.nii.gz'

        # subjects are detected as the nifti file names under src_dir
        self.expr = '^.resampled_distmap_(.*).nii.gz$'

        # Creates json log class
        json_file = join(self.crop_dir, self.side + 'distmap.json')
        self.json = LogJson(json_file)

        # Creates npys file name
        self.file_basename_npy = self.side + 'distmap'
        self.file_basename_pickle = self.side + 'distmap'

        self.input_type = 'distmap'


def parse_args(argv):
    """Function parsing command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        params: dictionary with keys: src_dir, tgt_dir, nb_subjects, list_sulci
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description='Generates cropped and npy files')
    parser.add_argument(
        "-s", "--src_dir", type=str, default=_RESAMPLED_SKELETON_DIR_DEFAULT,
        help='Source directory where input files lie. '
             'Input files are not cropped and represent a full hemisphere. '
             'They can be skeletons, labels, distance maps,... '
             'Default is : ' + _RESAMPLED_SKELETON_DIR_DEFAULT)
    parser.add_argument(
        "-y", "--input_type", type=str, default=_INPUT_TYPE_DEFAULT,
        help='Input type: \'skeleton\', \'foldlabel\', \'distmap\' '
        'Default is : ' + _INPUT_TYPE_DEFAULT)
    parser.add_argument(
        "-o", "--output_dir", type=str, default=_CROP_DIR_DEFAULT,
        help='Output directory where to store the cropped files. '
             'Default is : ' + _CROP_DIR_DEFAULT)
    parser.add_argument(
        "-k", "--mask_dir", type=str, default=_MASK_DIR_DEFAULT,
        help='masking directory where mask has been stored. '
             'Default is : ' + _MASK_DIR_DEFAULT)
    parser.add_argument(
        "-d", "--dilation", type=float, default=_DILATION_DEFAULT,
        help='Dilation size of mask. '
             'Default is : ' + str(_DILATION_DEFAULT))
    parser.add_argument(
        "-t", "--threshold", type=float, default=_THRESHOLD_DEFAULT,
        help='Threshold value of mask. '
             'Default is : ' + str(_THRESHOLD_DEFAULT))
    parser.add_argument(
        "-b", "--bbox_dir", type=str, default=_BBOX_DIR_DEFAULT,
        help='Bounding box directory where json files containing '
             'bounding box coordinates have been stored. '
             'Default is : ' + _BBOX_DIR_DEFAULT)
    parser.add_argument(
        "-u", "--sulcus", type=str, default=_SULCUS_DEFAULT, nargs='+',
        help='Sulcus name around which we determine the bounding box. '
             'If there are several sulci, add all sulci '
             'one after the other. Example: -u sulcus_1 sulcus_2 '
             'Default is : ' + _SULCUS_DEFAULT)
    parser.add_argument(
        "-i", "--side", type=str, default=_SIDE_DEFAULT,
        help='Hemisphere side (either L or R). Default is : ' + _SIDE_DEFAULT)
    parser.add_argument(
        "-a", "--parallel", default=False, action='store_true',
        help='if set (-a), launches computation in parallel')
    parser.add_argument(
        "-n", "--nb_subjects", type=str, default="all",
        help='Number of subjects to take into account, or \'all\'. '
             '0 subject is allowed, for debug purpose.'
             'Default is : all')
    parser.add_argument(
        "-c", "--cropping_type", type=str, default=_CROPPING_TYPE_DEFAULT,
        help='Method to select and crop the image. '
             'Type of cropping: '
             'bbox: for bounding box cropping'
             'mask: selection based on a mask'
             'mask_intersect: selection based on intersect of masks'
             'Default is : ' + _CROPPING_TYPE_DEFAULT)
    parser.add_argument(
        "-m", "--combine_type", type=bool, default=_COMBINE_TYPE_DEFAULT,
        help='Whether use specific combination of masks or not')
    parser.add_argument(
        "-p", "--no_mask", type=bool, default=_NO_MASK_DEFAULT,
        help='Whether apply mask')
    parser.add_argument(
        '-v', '--verbose', action='count', default=0,
        help='Verbose mode: '
             'If no option is provided then logging.INFO is selected. '
             'If one option -v (or -vv) or more is provided '
             'then logging.DEBUG is selected.')

    params = {}

    args = parser.parse_args(argv)

    # Writes command line argument to target dir for logging
    setup_log(
        args,
        log_dir=f"{args.output_dir}",
        prog_name=basename(__file__),
        suffix=f"right_{args.input_type}" if args.side == 'R'
               else f"left_{args.input_type}"
    )

    params = vars(args)

    params['crop_dir'] = args.output_dir
    params['list_sulci'] = args.sulcus  # a list of sulci

    # Checks if nb_subjects is either the string "all" or a positive integer
    params['nb_subjects'] = get_number_subjects(args.nb_subjects)

    # Removes renamed params
    # So that we can use params dictionary directly as function arguments
    params.pop('output_dir')
    params.pop('sulcus')
    params.pop('verbose')

    return params


def generate_crops(
        src_dir=_RESAMPLED_SKELETON_DIR_DEFAULT,
        input_type=_INPUT_TYPE_DEFAULT,
        crop_dir=_CROP_DIR_DEFAULT,
        bbox_dir=_BBOX_DIR_DEFAULT,
        mask_dir=_MASK_DIR_DEFAULT,
        side=_SIDE_DEFAULT,
        list_sulci=_SULCUS_DEFAULT,
        nb_subjects=_ALL_SUBJECTS,
        cropping_type=_CROPPING_TYPE_DEFAULT,
        combine_type=_COMBINE_TYPE_DEFAULT,
        parallel=False,
        no_mask=True,
        threshold=_THRESHOLD_DEFAULT,
        dilation=_DILATION_DEFAULT,
        njobs=_NB_JOBS_DEFAULT

):

    # Gets function arguments and values
    params = locals()
    params.pop('nb_subjects')
    params.pop('input_type')

    if input_type == "skeleton":
        crop = SkeletonCropGenerator(**params)
    elif input_type == "foldlabel":
        crop = FoldLabelCropGenerator(**params)
    elif input_type == "extremities":
        crop = ExtremitiesCropGenerator(**params)
    elif input_type == "distmap":
        crop = DistMapCropGenerator(**params)
    else:
        raise ValueError(
            "input_type: shall be either 'skeleton', 'foldlabel', "
            "'extremities' or 'distmap'")

    crop.compute(nb_subjects=nb_subjects)

    if input_type == "foldlabel":
        quality_checks(crop_dir, side)
    elif input_type == "extremities":
        quality_checks_extremities(crop_dir, side)


@exception_handler
def main(argv):
    """Reads argument line and creates cropped files and npy file
    Args:
        argv: a list containing command line arguments
    """

    # Parsing arguments
    params = parse_args(argv)

    # Actual API
    generate_crops(**params)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
