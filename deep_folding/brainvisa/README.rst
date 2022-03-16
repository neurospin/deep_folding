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
   # Hemisphere side
   SIDE=R
   # Output voxel size
   VOXEL_SIZE=1.0
   # Folder in which to write the results
   # Note that bounding bpxes will be written in the subfolder $SIDE
   BBOX_DIR=/neurospin/dico/data/deep_folding/test/bbox/${VOXEL_SIZE}mm
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

* $${SULCUS}_${SIDE}.json: a json file in the subfolder '/R' or '/L' (depending on the side) that contains the bounding box coordinates
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
   MASK_DIR=/neurospin/dico/data/deep_folding/test/bbox/${VOXEL_SIZE}mm

We then compute the mask:

.. code-block:: shell

  python3 compute_mask.py -s $SRC_DIR_SUPERVISED -o $MASK_DIR -u $SULCUS -i $SIDE -p $PATH_TO_GRAPH -x $VOXEL_SIZE

This will create in the folder $MASK_DIR four files 
(SIDE is now either 'left' or 'right'):

* ${SULCUS}_${SIDE}.nii.gz: a nifti file (and the header *.minf), in the subfolder '/R' or '/L' (depending on the side). This is the actual mask
* command_line_${SULCUS}_${SIDE}.sh: a bash file to reproduce the results (to be launched from deep_folding/brainvisa) 
* log_${SULCUS}_${SIDE}.log: a log file that contains the log of the command
