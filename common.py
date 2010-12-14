#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
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

MAJOR = 0
MINOR = 3
MICRO = 3
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
