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