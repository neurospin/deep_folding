#!/usr/bin/env python
# coding: utf-8
#
# Python afile auto-generated from notebooks/use_transform.py
# Creates transformation files This
# notebook creates the transformation files from raw MRI space to normalized
# SPM space It is auto-generated from the notebook

# # Imports

import sys
import os

# The following line permits to import deep_folding, even if this example is
# executed from the examples subfolder (and no install has been launched):
# 
#  /notebooks/use_transform.ipynb  
#  /deep_folding/__init__.py

sys.path.append((os.path.abspath('../')))
import deep_folding
from deep_folding.anatomist_tools import transform
print(os.path.dirname(deep_folding.__file__))

# # User-specific variables
# We now assign path names and other user-specific variables.
# The source directory is where the database lies.
# It contains the morphologist analysis subfolder ANALYSIS/3T_morphologist
#

src_dir = os.path.join(os.getcwd(), '../data/src')
src_dir = os.path.abspath(src_dir)
print("src_dir = " + src_dir)

# The target directory tgt_dir is where the files will be saved

tgt_dir = os.path.join(os.getcwd(), '../data/tgt')
tgt_dir = os.path.abspath(tgt_dir)
print("tgt_dir = " + tgt_dir)

print(sys.argv)

#####################################
# # Illustration of main program uses
#####################################

print(transform.__file__)

# With number of cells set to 0
args = "-n 0"
argv = args.split(' ')

transform.main(argv)

# Testig the help function
args = "--help"
argv = args.split(' ')

transform.main(argv)


# ### By using the API function call

transform.transform_to_spm(src_dir=src_dir, 
                       tgt_dir=tgt_dir, 
                       number_subjects=0)


# # Test example with all subjects from source directory

ref_dir = os.path.join(os.getcwd(), '../data/ref')
ref_dir = os.path.abspath(ref_dir)
print("ref_dir = " + ref_dir)


transform.transform_to_spm(src_dir=src_dir, tgt_dir=tgt_dir,
                           number_subjects=transform._ALL_SUBJECTS)

#####################################
# # Result analysis
#####################################

# Prints the list of files of the target directory

print('\n'.join(os.listdir(tgt_dir)))


# Expected output (we read the transformation file from the ref directory):

ref_file = os.listdir(ref_dir)[0]
print "ref_file = ", ref_file, '\n'
with open(os.path.join(ref_dir,ref_file), 'r') as f:
    print(f.read())


# Obtained output (we read the transformation file from the target directory):

tgt_file = os.listdir(tgt_dir)[0]
print"tgt_file = ", tgt_file, '\n'
with open(os.path.join(tgt_dir,tgt_file), 'r') as f:
    print(f.read())

# Generated README (we read the generated README from the target directory)

with open(os.path.join(tgt_dir,"README"), 'r') as f:
    print(f.read())

