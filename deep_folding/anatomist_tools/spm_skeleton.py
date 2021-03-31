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
The aim of this script is to compute transformation files
"""
from soma import aims
import os

_ALL_SUBJECTS = -1

_src_dir_default = "/neurospin/hcp/"
_tgt_dir_default = "/neurospin/dico/deep_folding_data/data/transfo_pre_process/"


root_dir_1 = "/neurospin/hcp/ANALYSIS/3T_morphologist/"
root_dir_2 = "/t1mri/default_acquisition/"


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

    def calculate_one_transform(subject_id):
        """Calculates the transformation file of a given subject.

        This transformation enables to go from native space (= MRI subject space) to
        normalized SPM space.

        Args:
            subject_id: int or str, id of subject of whom transformation file is computed
        """
        # Directory where subjects are stored
        root_dir = "/neurospin/hcp/ANALYSIS/3T_morphologist/" + str(subject_id) + "/t1mri/default_acquisition/"
        # Directory where to store transformation files
        saved_folder = "/neurospin/dico/lguillon/skeleton/transfo_pre_process/"
        # Transformation file that goes from native space to MNI space
        natif_to_mni = aims.read(root_dir + 'registration/RawT1-'+subject_id+'_default_acquisition_TO_Talairach-MNI.trm')

        # Fetching of template's transformation
        template = aims.read(root_dir + 'normalized_SPM_'+subject_id+'.nii')
        mni_to_template = aims.AffineTransformation3d(template.header()['transformations'][0]).inverse()
        # print(template.header()['transformations'][0])

        # Combination of transformations
        natif_to_template_mni = mni_to_template * natif_to_mni
        # print(natif_to_template_mni)

        # Saving of transformation files
        aims.write(natif_to_template_mni, self.tgt_dir + 'natif_to_template_spm_'+subject_id+'.trm')


    def calculate_transforms(self, number_subjects=_ALL_SUBJECTS):
        """Calculates transformation file for all subjects.

        This transformation enables to go from native space (= MRI subject space) to
        normalized SPM space.

        Args:
            number_subjects: integer giving the number of subjects,
                by default it is set to _ALL_SUBJECTS (-1).
        """

        # subjects are detected as the repertory name under src_dir
        list_all_subjects = os.listdir(self.src_dir)

        # Gives the possibility to list only the first number_subjects if requested
        list_subjects = ( list_all_subjects if number_subjects == _ALL_SUBJECTS
                          else list_all_subjects[:number_subjects])

        # Computes and saves transformation files for all listed subjects
        for subject in list_subjects:
            self.calculate_one_transform(subject)


def main():
    """

    """

    parser = argparse.ArgumentParser(prog='spm_skeleton',
                                     description='Generate transformation files')
    parser.add_argument("-s", "--src_dir", type=str, default=_src_dir_default,
                        help='Source directory where the MRI data lies. ' +
                             'Default is : ' + _src_dir_default)
    parser.add_argument("-t", "--tgt_dir", type=str, default=_tgt_dir_default,
                        help='Target directory where to store the output transformation files. ' +
                             'Default is : ' + _tgt_dir_default)
    parser.add_argument("-n", "--nb_subjects", type=int, default=_ALL_SUBJECTS,
                        help='(int) Number of subjects to take into account.' +
                             'Default is : ' + str(_ALL_SUBJECTS) + ' (all subjects)')
    args = parser.parse_args()
    src_dir = args.src_dir
    tgt_dir=args.tgt_dir
    nb_subjects=args.nb_subjects

    t = Transform(src_dir=src_dir, tgt_dir=tgt_dir)
    t.calculate_transforms(number_subjects=number_subjects)


######################################################################
# Main program
######################################################################

if __name__ == '__main__':
    main()
