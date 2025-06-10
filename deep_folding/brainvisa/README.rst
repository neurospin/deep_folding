deep_folding.brainvisa
######################

This folder contains scripts that work with PyAIMS (a BrainVISA library).

.. image:: ../../docs/general_scheme.png
  :width: 600

Creates an output directory and a json file
===========================================
Create an output directory for your dataset, named hereafter OUTPUT_DIR

.. code-block:: shell

    mkdir OUTPUT_DIR

In OUTPUT_DIR, create a json file named pipeline_loop_2mm.json, which contains:

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
        "labeled_subjects_dir": "/neurospin/dico/data/bv_databases/human/manually_labeled/pclean/all",
        "path_to_graph_supervised": "t1mri/t1/default_analysis/folds/3.3/base2018_manual",
        "supervised_output_dir": "/neurospin/dico/data/deep_folding/current",
        "nb_subjects_mask": -1,
        "graphs_dir": "/mnt/n4habcd",
        "path_to_graph": "ses-1/anat/t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_to_skeleton_with_hull": "ses-1/anat/t1mri/default_acquisition/default_analysis/segmentation",
        "skel_qc_path": "/neurospin/dico/data/deep_folding/current/datasets/ABCD/qc.tsv",
        "output_dir": "/neurospin/dico/data/deep_folding/current/datasets/ABCD",
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
