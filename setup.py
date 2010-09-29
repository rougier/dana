#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------
from common import *

if __name__ == '__main__':
    from distutils.core import setup
    setup(name             = DISTNAME,
          version          = VERSION,
          url              = URL,
          license          = LICENSE,
          author           = AUTHOR,
          author_email     = AUTHOR_EMAIL,
          description      = DESCRIPTION,
          long_description = LONG_DESCRIPTION,
          classifiers      = CLASSIFIERS,
          platforms        = PLATFORM,
          packages         =  ['dana',
                               'dana.tests'])
