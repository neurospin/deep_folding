
Deep folding
------------

The project aims to study cortical folding patterns thanks to deep learning tools.
MRIs are processed through BrainVISA/Morphologist tools.

Prerequisites
-------------

Anatomist parts (anatomist_tools) must run with brainvisa installed and python2

Deep learning part (preprocessing and utils) must run with python3 and works with pytorch.


Development
-----------

.. code-block:: shell

    git clone https://github.com/neurospin/deep_folding.git

    # Install for development
    bv bash
    cd deep_folding
    virtualenv -p /casa/install/bin/python venv
    ln -s /usr/local/lib/python2.7/dist-packages/sip.so venv/lib/python2.7/site-packages/sip.so
    ln -s /usr/local/lib/python2.7/dist-packages/PyQt5 venv/lib/python2.7/site-packages/PyQt5 
    . bin/activate/venv
    pip install -e .

    # Tests
    python -m pytest  # run tests



If you want to install the package:

.. code-block:: shell

    python setup.py install

Notebooks are in the repertory notebooks, access using:

.. code-block:: shell

    bv bash # to enter brainvisa environnment
    jupyter notebook # then click on file to open a notebook

Examples are in the examples directory

.. code-block:: shell

    bv bash
    cd examples
    python use_transform.py
