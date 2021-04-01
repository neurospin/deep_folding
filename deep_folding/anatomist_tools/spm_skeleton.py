# /usr/bin/env python2.7 + brainvisa compliant env
# coding: utf-8
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
The aim of this script is to compute transformation files from native to normalized SPM space



Examples:
    Specify the source directory where the MRI data lies, the target directory where to put the transform,
    and the number of subjects to analyse (all by default)
        $ python spm_skeleton.py -s /neurospin/hcp -t /neurospin/dico/deep_folding_data/data/transfo_pre_process -n all

Attributes:

"""
from soma import aims
from os import listdir
from os.path import join
import argparse

_ALL_SUBJECTS = -1

_src_dir_default = "/neurospin/hcp"
_tgt_dir_default = "/neurospin/dico/deep_folding_data/data/transfo_pre_process"


class Transform:
    """Compute transformation from native to normalized SPM space

    Attributes:
        src_dir: A string giving the name of the source data directory.
        tgt_dir: A string giving the name of the target directory in which the transformations are saved.
    """

    def __init__(self, src_dir=_src_dir_default, tgt_dir=_tgt_dir_default):
        """Inits Transform class with source and target directory names

        Args:
            src_dir: string naming src directory
            tgt_dir: string naming target directory in which transformtion files are saved
        """
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir

        # Subdirectories and files from the morphologist pipeline
        # Once the database directory (like /neurospin/hcp) is defined, the subdirectories remain identical
        # and are specific of the morphologist pipeline
        # 'subject' is the ID of the subject
        self.morphologist_dir = join(self.src_dir, "ANALYSIS/3T_morphologist")  # Morphologist directory
        self.morpho_acquis_dir = "%(subject)s/t1mri/default_acquisition"  # default acquisition subdirectory
        self.normalized_spm_file = "normalized_SPM_%(subject)s.nii"  # (input) name of normalized SPM file
        # (input) name of the raw to MNT Talairach transformation file
        self.transform_to_talairach_file = "registration/RawT1-%(subject)s_default_acquisition_TO_Talairach-MNI.trm"

        # (Output files) Name of transformation files that are written by the program
        # 'subject' is the ID of the subject
        self.transform_to_spm_file = "natif_to_template_spm_%(subject)s.trm"

    def calculate_one_transform(self, subject_id):
        """Calculates the transformation file of a given subject.

        This transformation enables to go from native space (= MRI subject space) to
        normalized SPM space.

        Args:
            subject_id: int or str, id of subject of whom transformation file is computed
        """

        # Identifies 'subject' in a mapping (for file and directory namings)
        subject = {'subject': subject_id}

        # Names directory where subject analysis files are stored
        subject_dir = join(self.morphologist_dir, self.morpho_acquis_dir % subject)
        # Reads transformation file that goes from native space to MNI space
        natif_to_mni = aims.read(join(subject_dir, self.transform_to_talairach_file % subject))

        # Fetching of template's transformation
        template = aims.read(join(subject_dir, self.normalized_spm_file % subject))
        mni_to_template = aims.AffineTransformation3d(template.header()['transformations'][0]).inverse()
        # print(template.header()['transformations'][0])

        # Combination of transformations
        natif_to_template_mni = mni_to_template * natif_to_mni
        # print(natif_to_template_mni)

        # Saving of transformation files
        transform_to_spm_file = join(self.tgt_dir, self.transform_to_spm_file % subject)
        aims.write(natif_to_template_mni, transform_to_spm_file)

    def calculate_transforms(self, number_subjects=_ALL_SUBJECTS):
        """Calculates transformation file for all subjects.

        This transformation enables to go from native space (= MRI subject space) to
        normalized SPM space.

        Args:
            number_subjects: integer giving the number of subjects,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        # subjects are detected as the repertory name under src_dir
        list_all_subjects = listdir(self.morphologist_dir)

        # Gives the possibility to list only the first number_subjects if requested
        list_subjects = (list_all_subjects if number_subjects == _ALL_SUBJECTS
                         else list_all_subjects[:number_subjects])

        # Computes and saves transformation files for all listed subjects
        for subject in list_subjects:
            print("subject : " + subject)
            self.calculate_one_transform(subject)


def main():
    """

    """

    # Parse arguments
    parser = argparse.ArgumentParser(prog='spm_skeleton',
                                     description='Generate transformation files')
    parser.add_argument("-s", "--src_dir", type=str, default=_src_dir_default,
                        help='Source directory where the MRI data lies. '
                             'Default is : ' + _src_dir_default)
    parser.add_argument("-t", "--tgt_dir", type=str, default=_tgt_dir_default,
                        help='Target directory where to store the output transformation files. '
                             'Default is : ' + _tgt_dir_default)
    parser.add_argument("-n", "--nb_subjects", type=int, default=_ALL_SUBJECTS,
                        help='(int) Number of subjects to take into account.'
                             'Default is : ' + str(_ALL_SUBJECTS) + ' (all subjects)')
    args = parser.parse_args()
    src_dir = args.src_dir
    tgt_dir = args.tgt_dir
    number_subjects = args.nb_subjects

    # Check if nb_subjects is either the string "all" or a positive integer
    try:
        if number_subjects == "all":
            number_subjects = _ALL_SUBJECTS
        else:
            number_subjects = int(number_subjects)
            if number_subjects <= 0:
                raise ValueError
    except ValueError:
        raise ValueError("nb_subjects must be either the string \"all\" or an integer")

    t = Transform(src_dir=src_dir, tgt_dir=tgt_dir)
    t.calculate_transforms(number_subjects=number_subjects)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    main()
