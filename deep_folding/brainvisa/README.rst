deep_folding.brainvisa
######################

This folder contains scripts that work with BrainVISA/Anatomist.

.. image:: ../../docs/general_scheme.png
  :width: 600


Step-by-step tutorial: generate a dataset
#########################################

We start by computing, for each sulcus, bounding boxes and masks.
For this, we are using a manually labelled database. At the end of the programs,
we will have a bounding box that encompasses the sulcus with the given name
for all subjects. In the same way, we will obtain a mask that encompasses
the given sulcus for all subjects.

Compute a sulcus-specific bounding box
======================================

We suppose that we have already installed brainvisa singularity image 
and the deep_folding module following the steps described in `<../../README.rst>`_

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
   BBOX_DIR=/neurospin/dico/data/deep_folding/test/bbox/${VOXEL_SIZE}mm/${SIDE}
   # sulcus name (without the _left or _right extension)
   SULCUS=F.C.M.ant.
   # Relative path of graph in the folder hierarchy of the subject
   PATH_TO_GRAPH=t1mri/t1/default_analysis/folds/3.3/base2018_manual


We then go to the deep_folding/brainvisa folder:

.. code-block:: shell

   cd deep_folding/brainvisa

We then determine the bounding box around the sulcus named SULCUS:

.. code-block:: shell

  python3 compute_bounding_box.py -s $SRC_DIR_SUPERVISED -b $BBOX_DIR -u $SULCUS -i $SIDE -p $PATH_TO_GRAPH -x $VOXEL_SIZE

This will create in the folder $BBOX_DIR three files 
(SIDE is now either 'left' or 'right'):

* ${SULCUS}_${SIDE}.json: a json file that contains the bounding box coordinates
* command_line_${SULCUS}_${SIDE}.sh: a bash file to reproduce the results (to be launched from deep_folding/brainvisa) 
* log_${SULCUS}_${SIDE}.log: a log file that contains the log of the command

Compute a sulcus-specific mask
==============================

We will now compiute the mask encompassing the sulcus SULCUS:

We first define a mask folder to put the results of the mask:

.. code-block:: shell

   # Folder in which to write the mask results
   MASK_DIR=/neurospin/dico/data/deep_folding/test/bbox/${VOXEL_SIZE}mm/${SIDE}

We then compute the mask:

.. code-block:: shell

  python3 compute_mask.py -s $SRC_DIR_SUPERVISED -m $BBOX_DIR -u $SULCUS -i $SIDE -p $PATH_TO_GRAPH -x $VOXEL_SIZE


This will create in the folder $MASK_DIR four files 
(SIDE is now either 'left' or 'right'):

* ${SULCUS}_${SIDE}.nii.gz: a nifti file (and ts header *.minf), which is the actual mask
* command_line_${SULCUS}_${SIDE}.sh: a bash file to reproduce the results (to be launched from deep_folding/brainvisa) 
* log_${SULCUS}_${SIDE}.log: a log file that contains the log of the command
