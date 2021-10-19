Anatomist Tools
###############

This folder contains scripts that work with BrainVISA/Anatomist.

Description of modules
======================

mask.py
------------------
Outputs the bounding box of a specific sulcus based on a manually labeled dataset.
Boundig boxes ared defined in the normalized SPM space

load_data.py
------------
Enables to create and to save to .pickle a dataframe of numpy arrays from a folder of .nii.gz or
.nii images.
Subjects IDs are kept.

display.py
----------
Diplays model's outputs (saved as numpy arrays) in an Anatomist window.

benchmark_generation.py
-----------------------
Creates altered skeletons (with some simple surfaces lacking) benchmark.

benchmark_pipeline.py
---------------------
Normalizes, resamples and crops the altered skeletons.

Tutorial: generate a dataset
============================


