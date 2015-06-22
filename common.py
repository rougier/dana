#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
descr = '''
DANA (Distributed (asynchronous) Numerical and Adaptive computing) is a python
computing framework based on numpy and scipy libraries whose primary goals
relate to computational neuroscience and artificial neural networks. However,
this framework can be used in several different areas like physic simulations,
cellular automata or image processing.

The computational paradigm supporting the DANA framework is grounded on the
notion of a unit that is a set of arbitrary values that can vary along time
under the influence of other units and learning. Each unit can be linked to any
other unit (including itself) using a weighted link and a group is a structured
set of such homogeneous units.
'''

DISTNAME            = 'dana'
DESCRIPTION         = 'Distributed (Asynchronous) Numerical Adaptive computing framework'
LONG_DESCRIPTION    = descr
AUTHOR              = 'Nicolas P. Rougier'
AUTHOR_EMAIL        = 'Nicolas.Rougier@loria.fr'
URL                 = 'http://dana.loria.fr'
LICENSE             = 'BSD'
DOWNLOAD_URL        = 'http://pypi.python.org/pypi/dana'
PLATFORM            = 'any'
REQUIRES            = ['numpy (>=1.5)']

MAJOR = 0
MINOR = 5
MICRO = 1
ALPHA = False
BETA  = False
DEV   = False

CLASSIFIERS = ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: BSD License',
               'Topic :: Scientific/Engineering']

def build_verstring():
    return '%d.%d.%d' % (MAJOR, MINOR, MICRO)

def build_fverstring():
    if DEV:
        return build_verstring() + '.dev'
    elif ALPHA:
        return build_verstring() + '.alpha'
    elif BETA:
        return build_verstring() + '.beta'
    else:
        return build_verstring()

def write_info(fname):
    f = open(fname, "w")
    f.writelines("# THIS FILE IS GENERATED FROM THE SETUP.PY. DO NOT EDIT.\n")
    f.writelines('"""%s"""\n' % descr)
    f.writelines("version = '%s'\n" % build_verstring())
    f.writelines("release = '%s'\n" % build_fverstring())
    f.close()

VERSION = build_fverstring()
INSTALL_REQUIRE = ['numpy','scipy']
write_info('dana/info.py')
