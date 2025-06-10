deep_folding.brainvisa
######################

This folder contains scripts that work with PyAIMS (a BrainVISA library).

.. image:: ../../docs/general_scheme.png
  :width: 600

Creates an output directory and a json file
===========================================
We refer to the path to the library deep_folding as PROGRAM_DIR.

We refer to the path that contains morphologist subjects as MORPHOLOGIST_DIR.

Create a directory for all your datasets, named hereafter DATASET_DIR.

Inside DATASET_DIR, create an output directory for your specific dataset (named DATASET). The full path is named hereafter OUTPUT_DIR (="DATASET_DIR/DATASET")

.. code-block:: shell

    mkdir OUTPUT_DIR

In OUTPUT_DIR, create a json file named pipeline_loop_2mm.json, which contains (replace OUTPUT_DIR and PROGRAM_DIR with the corresponding paths). For the key "path_top_graph", enter the relative path to the Morphologist grapĥ subdirectory. For the "path_top_skeleton_with_hull", enter the relative path to the segmentation directory of Morphologist.

.. code-block:: shell

  {
      "save_behavior": "best",
      "side": "L",
      "out_voxel_size": 2.0,
      "region_name": "S.C.-sylv.",
      "brain_regions_json": "PROGRAM_DIR/deep_folding/data/sulci_regions_gridsearch.json",
      "parallel": true,
      "nb_subjects": -1,
      "input_type": "extremities",
      "labeled_subjects_dir": "",
      "path_to_graph_supervised": "",
      "supervised_output_dir": "PROGRAM_DIR/deep_folding/data",
      "nb_subjects_mask": -1,
      "graphs_dir": "MORPHOLOGIST_DIR",
      "path_to_graph": "ses-1/anat/t1mri/default_acquisition/default_analysis/folds/3.1",
      "path_to_skeleton_with_hull": "ses-1/anat/t1mri/default_acquisition/default_analysis/segmentation",
      "skel_qc_path": "",
      "output_dir": "OUTPUT_DIR",
      "junction": "thin",
      "bids": false,
      "new_sulcus": null,
      "resampled_skel": false,
      "cropping_type": "mask",
      "combine_type": false,
      "no_mask": false,
      "threshold": 0,
      "dilation": 5
  }

multi_pipelines.py
==================
In the file multi_pipelines.py:

* Replace the value of path_dataset_root with DATASET_DIR.
* Replace the value of datasets with ["DATASET"]

Inside the brainvisa apptainer or inside the appropriate pixi environnement (see `<../../README.rst>`_), launch:

.. code-block:: shell

    python3 multi_pipelines.py





