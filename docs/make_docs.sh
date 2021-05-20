#!/bin/bash

BUILDDIR=../../deep_folding_docs

if [ ! -d $BUILDDIR/html ]; then
    cd source
    make firstbuild
    cd ..
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/host/usr/lib
export PYTHONPATH=$PYTHONPATH:/host/usr/lib/python2.7/lib-dynload:/usr/local/lib/python2.7

cd source
make buildandpush