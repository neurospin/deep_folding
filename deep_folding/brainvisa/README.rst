deep_folding.brainvisa
######################

This folder contains scripts that work with BrainVISA/Anatomist.

.. image:: ../../docs/general_scheme.png
  :width: 600


Step-by-step tutorial: generate a dataset
#########################################

The first part of the pipeline (first column of the figure) is using a manually labelled dataset.
In such a dataset, each sulcus is manually labelled. Names of the different sulci can be found as a json file at
`<https://github.com/brainvisa/brainvisa-share/blob/master/nomenclature/translation/sulci_default_list.json>`_, and they are represented in the brain here:

.. image:: https://brainvisa.info/web/_static/images/bsa/brainvisa_sulci_atlas_with_table_150dpi-r90.png
  :width: 600

The first step is to generate for each sulcus of interest a bounding box (compute_bounding_box.py) and a mask (compute_mask.py) that encompass the given sulcus for all subjects of the manually labelled dataset.

The second step is then to generate crops (generate_crops.py) in the target dataset (for example `HCP http://www.humanconnectomeproject.org/>`_) using different combinations of masks and/or bounding boxes. For visualization, we generate each crop as nifti file; in order to provide the deep learning program with an appropriate input, we combine all volumes in a dataframe and save it as a pickle file.

When generating crops, sulci can be regrouped in different overlapping regions. We have subdivided the brain in different overlapping regions as a json file at `<https://github.com/brainvisa/brainvisa-share/blob/master/nomenclature/translation/sulci_regions_overlap.json>`_.

Compute a sulcus-specific bounding box
======================================

We start by computing, for each sulcus, bounding boxes and masks.
For this, we are using a manually labelled database. At the end of the program,
we will have a bounding box that encompasses the sulcus with the given name
for all subjects. 

We suppose that we have already installed brainvisa singularity image 
as well as the deep_folding module following the steps described in `<../../README.rst>`_

We first need to enter in the brainvisa singularity bash:

.. code-block:: shell

   bv bash

We then define the parameters to launch the computation of the bounding box:

.. code-block:: shell

   # We change the following variable with our manually labelled dataset.
   # This one is the one used at Neurospin:
   SRC_DIR_SUPERVISED=/neurospin/dico/data/bv_databases/human/pclean/all
   
   # Global output directory where to store the files
   # We change this to the desired output directory
   OUTPUT_DIR=/neurospin/dico/data/deep_folding/test
   
   # Hemisphere side
   SIDE=R
   # Output voxel size
   VOXEL_SIZE=2.0
   # Folder in which to write the results
   # Note that bounding bpxes will be written in the subfolder $SIDE
   BBOX_DIR=${OUTPUT_DIR}/bbox/${VOXEL_SIZE}mm
   # sulcus name (without the _left or _right extension)
   SULCUS=F.C.M.ant.
   # Relative path of graph in the folder hierarchy of the subject
   PATH_TO_GRAPH=t1mri/t1/default_analysis/folds/3.3/base2018_manual


We then go to the deep_folding/brainvisa folder:

.. code-block:: shell

   cd deep_folding/brainvisa

We then determine the bounding box around the sulcus named SULCUS by launching the following command:

.. code-block:: shell

  python3 compute_bounding_box.py -s $SRC_DIR_SUPERVISED -o $BBOX_DIR -u $SULCUS -i $SIDE -p $PATH_TO_GRAPH -x $VOXEL_SIZE

This will create in the folder $BBOX_DIR three files 
(SIDE is now either 'left' or 'right'):

* ${SULCUS}_${SIDE}.json: a json file in the subfolder '/R' or '/L' (depending on the side) that contains the bounding box coordinates
* command_line_${SULCUS}_${SIDE}.sh: a bash file to reproduce the results (to be launched from deep_folding/brainvisa) 
* log_${SULCUS}_${SIDE}.log: a log file that contains the log of the command

We note that we can also change the sulcus name using the flag '-w'. Launch the help command for more information:

.. code-block:: shell

  python3 compute_bounding_box.py -h

Compute a sulcus-specific mask
==============================

We will now compute the mask encompassing the sulcus SULCUS:

We first define a mask folder to put the results of the mask:

.. code-block:: shell

   # Folder in which to write the mask results
   MASK_DIR=${OUTPUT_DIR}/bbox/${VOXEL_SIZE}mm

We then compute the mask:

.. code-block:: shell

  python3 compute_mask.py -s $SRC_DIR_SUPERVISED -o $MASK_DIR -u $SULCUS -i $SIDE -p $PATH_TO_GRAPH -x $VOXEL_SIZE

This will create in the folder $MASK_DIR four files 
(SIDE is now either 'left' or 'right'):

* ${SULCUS}_${SIDE}.nii.gz: a nifti file (and the header *.minf), in the subfolder '/R' or '/L' (depending on the side). This is the actual mask
* command_line_${SULCUS}_${SIDE}.sh: a bash file to reproduce the results (to be launched from deep_folding/brainvisa) 
* log_${SULCUS}_${SIDE}.log: a log file that contains the log of the command


Generate skeletons and foldlabels in the native space
=====================================================

We now generate skeletons and foldlabels from graph for the unsupervised target data set. Such a dataset can be for example the HCP database analyzed using morphologist.

We define relevant directories:

.. code-block:: shell

   # We change the following variable with the unsupervised dataset.
   # This variable points directly to the morphologist directory containing the subjects as subdirectories.
   # This one is the HCP dataset used at Neurospin:
   SRC_DIR_UNSUPERVISED=/neurospin/dico/data/bv_databases/human/hcp/hcp
   
   # Output directory where to put raw skeletons:
   SKELETON_DIR=${OUTPUT_DIR}/hcp/skeletons/raw
   
   # Output directory where to put raw foldlabels:
   FOLDLABEL_DIR=${OUTPUT_DIR}/hcp/foldlabels/raw
   
   # Relative path to graph for our HCP dataset
   PATH_TO_GRAPH_HCP=t1mri/BL/default_analysis/folds/3.1

We generate raw skeletons from graph, without resampling at this stage. Note the option '-a' that tells the program to parallelize computation. If the program fails, remove the option '-a' and add the option '-v' (verbose mode) to get more debug outputs:

.. code-block:: shell

    python3 generate_skeletons.py -s $SRC_DIR_UNSUPERVISED -o $SKELETON_DIR -i $SIDE -p $PATH_TO_GRAPH_HCP -a

In the same way, we generate raw foldlabel files:

.. code-block:: shell

    python3 generate_foldlabels.py -s $SRC_DIR_UNSUPERVISED -o $FOLDLABEL_DIR -i $SIDE -p $PATH_TO_GRAPH_HCP -a
  
Resample skeletons and foldlabels in the native space
=====================================================
 
We will now resample skeletons and foldlabels with the desired voxel size and in the ICBM2009c template. To avoid having to read graph files several times, we first compute the linear tranformation from the native space to the ICBM2009c space:

We first define the transform output directory where to store transform files, as well as the output directories where to store resampled skeleton and foldlabels files:

.. code-block:: shell

    # Output directory where to put transform files:
    TRANSFORM_DIR=${OUTPUT_DIR}/datasets/hcp/transforms

    # Output directories where to put resamples skeleton and foldlabels files:
    RESAMPLED_SKELETON_DIR=${OUTPUT_DIR}/datasets/hcp/skeletons/{VOXEL_SIZE}mm
    RESAMPLED_FOLDLABEL_DIR=${OUTPUT_DIR}/datasets/hcp/foldlabels/{VOXEL_SIZE}mm
  
We then generate transform files for the whole dataset:
 
.. code-block:: shell

    python3 generate_ICBM2009c_transforms.py -s $SRC_DIR_UNSUPERVISED -o $TRANSFORM_DIR -i $SIDE -p $PATH_TO_GRAPH_HCP -a
 
Using these transform files, we resample skeletons and foldlabels using resample_files.py:
 
.. code-block:: shell

    python3 resample_files.py -s $SKELETON_DIR -o $RESAMPLED_SKELETON_DIR -i $SIDE -y skeleton -t $TRANSFORM_DIR -x $VOXEL_SIZE -a
    python3 resample_files.py -s $FOLDLABEL_DIR -o $RESAMPLED_FOLDLABEL_DIR -i $SIDE -y foldlabel -t $TRANSFORM_DIR -x $VOXEL_SIZE -a
  
Crop generation
===============

We can now generate crops quickly as it is simply a mask or a crop of the resampled volume. We can combine sulci by just adding suylci in the sulcus list (see the help command for more information). We are now generating a crop based on the mask of a single sulcus named $SULCUS.

We first define the relevant output crop directory:

.. code-block:: shell

    CROP_DIR=${OUTPUT_DIR}/datasets/hcp/crops/${VOXEL_SIZE}mm/${SULCUS}/mask

We then generate skeleton crops and foldlabel crops. The effective mask is saved as ${SIDE}mask.nii.gz in the crop directory:

.. code-block:: shell

    python3 generate_crops.py -s $RESAMPLED_SKELETON_DIR -o $CROP_DIR -i $SIDE -y skeleton -k $MASK_DIR -u $SULCUS -c mask -a
    python3 generate_crops.py -s $RESAMPLED_SKELETON_DIR -o $CROP_DIR -i $SIDE -y foldlabel -k $MASK_DIR -u $SULCUS -c mask -a
  
 
