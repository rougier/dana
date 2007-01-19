#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage : build_install.sh install_dir"
    exit
fi
./setup1.py build
./setup1.py install --prefix=$1
./setup2.py build
./setup2.py install --prefix=$1
