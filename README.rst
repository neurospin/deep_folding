
Deep folding
------------

The project aims to study cortical folding patterns thanks to deep learning tools.
MRIs are processed through BrainVISA/Morphologist tools.

Prerequisites
-------------

Brainvisa parts (deep_folding.brainvisa) must run with brainvisa installer (usually using brainvisa singularity container).

Deep learning part (preprocessing and utils) must run with python3 and works with pytorch.

Package documentation can be found at `https://neurospin.github.io/deep_folding/index.html <https://neurospin.github.io/deep_folding/index.html>`_.

Generates datasets of crops
---------------------------

Deep learning pipelines to investigate folding patterns are not working on the whole brain (or the whole hemisphere) but on brain crops.
Several processings are required, as drawn here:

.. image:: docs/deep_folding.png
  :width: 1000
 
We give a step-by-step description of the pipeline in `<deep_folding/brainvisa/README.rst>`_.

Development
-----------

.. code-block:: shell

    git clone https://github.com/neurospin/deep_folding.git

    # Install for development
    bv bash
    cd deep_folding
    virtualenv --python=python3 --system-site-packages venv
    . venv/bin/activate
    pip3 install -e .

    # Tests
    python3 -m pytest  # run tests



If you want to install the package:

.. code-block:: shell

    python3 setup.py install

Notebooks are in the folder notebooks, access using:

.. code-block:: shell

    bv bash # to enter brainvisa environnment
    . venv/bin/activate
    jupyter notebook # then click on file to open a notebook

If you want to build the documentation and pushes it to the web:

.. code-block:: shell

    bv bash # to enter brainvisa environnment
    . venv/bin/activate
    pip3 install -e .[doc]
    cd docs
    ./make_docs.sh


If you want to clean the documentation:

.. code-block:: shell

    cd docs/source
    make clean

