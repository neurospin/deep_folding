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

"""
    Utilities to perform quality checks
"""

import glob
import re
import csv
import os

from deep_folding.config.logs import set_file_logger
# Defines logger
log = set_file_logger(__file__)


def compare_number_aims_files_with_expected(output_dir: str,
                                            list_subjects: list):
    """Compares number of generated files and expected number"""

    all_files = glob.glob(f"{output_dir}/*")

    generated_files = [f for f in all_files
                       if not re.search('.minf$', f)]
    log.debug(f"Output directory = {output_dir}")
    log.debug(f"Generated_files = {generated_files}")

    nb_generated_files = len(generated_files)
    nb_expected_files = len(list_subjects)

    log.info(f"\n\tNumber of generated files = {nb_generated_files}")
    log.info(f"\n\tNumber of requested files = {nb_expected_files}")

    if nb_generated_files != nb_expected_files:
        log.warning("Number of generated files != number of requested files "
                    "as determined by number of subjects")


def compare_number_aims_files_with_number_in_source(output_dir: str,
                                                    src_dir: str):
    """Compares number of generated files and source files"""

    all_files = glob.glob(f"{output_dir}/*")

    generated_files = [f for f in all_files
                       if not re.search('.minf$', f)]
    log.debug(f"Output directory = {output_dir}")
    log.debug(f"Generated_files = {generated_files}")

    src_files = glob.glob(f"{src_dir}/*")

    src_files = [f for f in src_files
                 if not re.search('.minf$', f)]

    nb_generated_files = len(generated_files)
    nb_expected_files = len(src_files)

    log.info(f"\n\tNumber of generated files = {nb_generated_files}")
    log.info(f"\n\tNumber of source files = {nb_expected_files}")

    if nb_generated_files != nb_expected_files:
        log.warning("Number of generated files != number of source files. "
                    "This is the important warning to look at "
                    "if you want to process the whole dataset")

    return generated_files, src_files


def get_not_processed_files(src_dir, tgt_dir, src_filename):
    """Returns list of source files not yet processed.

    This is done by comparing subjects in src and tgt directories"""

    if isinstance(src_dir, str):
        src_files = glob.glob(f"{src_dir}/*.nii.gz")
    log.info(f"number of source files = {len(src_files)}")
    if len(src_files):
        log.info(f"first source file = {src_files[0]}")
    log.debug(f"list src files = {src_files}")

    tgt_files = glob.glob(f"{tgt_dir}/*.nii.gz")
    log.info(f"number of target files = {len(tgt_files)}")
    if len(tgt_files):
        log.info(f"first target file = {tgt_files[0]}")
        tgt_subjects = [subject.split("resampled_")[-1]
                        for subject in tgt_files]
        tgt_subjects = ['_'.join(subject.split("_")[1:])
                        for subject in tgt_subjects]
        tgt_subjects = [subject.split(".")[0] for subject in tgt_subjects]
        log.info(f"first target subject = {tgt_subjects[0]}")
    else:
        tgt_subjects = []

    src_subjects = [subject.split(src_filename)[-1] for subject in src_files]
    log.info("src subjects before . split: " + src_subjects[0])
    src_subjects = [subject.split(".")[0] for subject in src_subjects]
    log.info("Src subjects after . split: " + src_subjects[0])

    not_processed_subjects = list(set(src_subjects) - set(tgt_subjects))

    root = src_files[0].split(src_filename)[0]
    log.info("src_filename: " + src_filename)
    log.info("root: " + root)
    not_processed_files = [
        f"{root}{src_filename}{subject}.nii.gz"
        for subject in not_processed_subjects]
    log.info(f"number of not processed subjects = {len(not_processed_files)}")
    if len(not_processed_files):
        log.info(f"first not_processed file = {not_processed_files[0]}")

    return not_processed_files


def get_not_processed_subjects_dict(subjects, tgt_dir):
    """Returns list of subjects not yet processed.

    This is done by comparing subjects in subject dict and tgt directories"""

    log.info(f"first subject start of fucntion= {subjects[0]}")
    src_subjects = [sub['subject'] for sub in subjects]
    if len(src_subjects):
        log.info(f"first subject = {src_subjects[0]}")

    tgt_files = glob.glob(f"{tgt_dir}/*.nii.gz")
    log.info(f"number of target files = {len(tgt_files)}")
    if len(tgt_files):
        log.info(f"first target file = {tgt_files[0]}")

    tgt_subjects = [os.path.basename(file) for file in tgt_files]
    tgt_subjects = [subject.split(".")[0] for subject in tgt_subjects]

    tgt_subjects = [subject.split(".")[0] for subject in tgt_subjects]

    not_processed_subjects = list(set(src_subjects) - set(tgt_subjects))

    not_processed_subjects_dict = []
    for sub in not_processed_subjects:
        for s in subjects:
            if s['subject'] == sub:
                not_processed_subjects_dict.append(s)

    log.info(
        f"number of not processed subjects = "
        f"{len(not_processed_subjects_dict)}")
    if len(not_processed_subjects):
        log.info(
            f"first not_processed subject = {not_processed_subjects_dict[0]}")

    return not_processed_subjects_dict


def get_not_processed_cropped_files(src_dir, tgt_dir):
    """Returns list of source files not yet processed.

    this one is specific toc rop directory.
    This is done by comparing subjects in src and tgt directories"""

    if isinstance(src_dir, str):
        src_files = glob.glob(f"{src_dir}/*.nii.gz")
    log.info(f"number of source files = {len(src_files)}")
    if len(src_files):
        log.info(f"first source file = {src_files[0]}")
    log.debug(f"list src files = {src_files}")

    tgt_files = glob.glob(f"{tgt_dir}/*.nii.gz")
    log.info(f"number of target files = {len(tgt_files)}")
    if len(tgt_files):
        log.info(f"first target file = {tgt_files[0]}")

    src_subjects = [subject.split("resampled_")[-1] for subject in src_files]
    src_subjects = ['_'.join(subject.split("_")[1:])
                    for subject in src_subjects]
    src_subjects = [subject.split(".")[0] for subject in src_subjects]

    tgt_subjects = [subject.split("_cropped")[0] for subject in tgt_files]
    tgt_subjects = [subject.split("/")[-1] for subject in tgt_subjects]

    not_processed_subjects = list(set(src_subjects) - set(tgt_subjects))

    root = src_files[0].split("resampled_")[0]
    root2 = src_files[0].split("resampled_")[1].split('_')[0]
    not_processed_files = [
        f"{root}resampled_{root2}_{subject}.nii.gz"
        for subject in not_processed_subjects]
    log.info(f"number of not processed subjects = {len(not_processed_files)}")
    if len(not_processed_files):
        log.info(f"first not_processed file = {not_processed_files[0]}")

    return not_processed_files


def get_not_processed_subjects(src_subjects, tgt_dir, prefix="generated_"):
    """Returns list of source files not yet processed.

    This is done by comparing subjects in src and tgt directories"""

    log.info(f"number of source subjects = {len(src_subjects)}")
    if len(src_subjects):
        log.info(f"first src subject = {src_subjects[0]}")
    tgt_files = glob.glob(f"{tgt_dir}/*.nii.gz")
    log.info(f"number of target files = {len(tgt_files)}")
    if len(tgt_files):
        log.info(f"first target file = {tgt_files[0]}")

    tgt_subjects = [subject.split(prefix)[-1] for subject in tgt_files]
    tgt_subjects = [subject.split("_")[0] for subject in tgt_subjects]

    tgt_subjects = [subject.split(".")[0] for subject in tgt_subjects]

    if len(tgt_subjects):
        log.info(f"first tgt subject = {tgt_subjects[0]}")

    not_processed_subjects = list(set(src_subjects) - set(tgt_subjects))

    return not_processed_subjects


def get_not_processed_subjects_distmap(
        src_subjects, tgt_dir, prefix="generated_"):
    """Returns list of source files not yet processed.

    This is done by comparing subjects in src and tgt directories"""

    log.info(f"number of source subjects = {len(src_subjects)}")
    if len(src_subjects):
        log.info(f"first src subject = {src_subjects[0]}")
    tgt_files = glob.glob(f"{tgt_dir}/*.nii.gz")
    log.info(f"number of target files = {len(tgt_files)}")
    if len(tgt_files):
        log.info(f"first target file = {tgt_files[0]}")

    tgt_subjects = [subject.split(prefix)[-1] for subject in tgt_files]
    # tgt_subjects = [subject.split("_")[0] for subject in tgt_subjects]

    tgt_subjects = [subject.split(".")[0] for subject in tgt_subjects]

    if len(tgt_subjects):
        log.info(f"first tgt subject = {tgt_subjects[0]}")

    not_processed_subjects = list(set(src_subjects) - set(tgt_subjects))
    over_processed_subjects = list(set(tgt_subjects) - set(src_subjects))
    log.info(f"Over processed subjects = {over_processed_subjects}")

    return not_processed_subjects


def get_not_processed_subjects_transform(
        src_subjects, tgt_dir, prefix="ICBM2009c_"):
    """Returns list of source files not yet processed.

    This is done by comparing subjects in src and tgt directories"""

    log.info(f"number of source subjects = {len(src_subjects)}")
    if len(src_subjects):
        log.info(f"first subject = {src_subjects[0]}")
    tgt_files = glob.glob(f"{tgt_dir}/*.trm")
    log.info(f"number of target files = {len(tgt_files)}")
    if len(tgt_files):
        log.info(f"first target file = {tgt_files[0]}")

    tgt_subjects = [subject.split(prefix)[-1] for subject in tgt_files]
    tgt_subjects = [subject.split("_")[0] for subject in tgt_subjects]

    tgt_subjects = [subject.split(".")[0] for subject in tgt_subjects]

    not_processed_subjects = list(set(src_subjects) - set(tgt_subjects))

    return not_processed_subjects


def save_list_to_csv(not_processed_files, csv_file_name):
    """Saves list of not_processed files to csv"""

    list_of_lists = [[e] for e in not_processed_files]
    with open(csv_file_name, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(list_of_lists)
