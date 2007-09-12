#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright (c) 2006-2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id: __init__.py 166 2007-05-18 07:54:33Z rougier $
#------------------------------------------------------------------------------
""" core component

From a conceptual point of view, the computational paradigm supporting the
library is grounded on the notion of a unit that is essentially a potential
that can vary along time under the influence of other units and learning.
Those units are organized into layers, maps and network: a network is made
of one to several map, a map is made of one to several layer and a layer is
made of a set of several units. Each unit can be linked to any other unit
(included itself) using a weighted link.
"""

import time
import threading
import sys, os.path
from _core import *
import random

__all__ = ['Model','Network', 'Environment', 'Map', 'Layer', 'Unit', 
           'Link', 'Event', 'EventDP', 'EventDW', 'EventEvaluate',
           'Observer', 'Observable']

_random = Random()

def seed (s = None):
    """ Seed for internal random generator """

    if not s:
        random.seed (_random.seed)
        return _random.seed
    random.seed (s)
    _random.seed = s
    return _random.seed

seed (12345)


Object.__write = Object.write
def __write (self, filename):
    """ """
    co = sys._getframe(1).f_code
    script = "\n"
    f = file (co.co_filename, "r")
    for line in f:
        script += line
    f.close()
    if filename[-3:] != '.gz':
        filename += '.gz'
    self.__write (filename,  os.path.abspath(co.co_filename), script)
Object.write = __write


# ___________________________________________________________________ModelThread
class ModelThread (threading.Thread):
    """
    Model Thread class
    """

    def __init__ (self, model, n, block):
        """

        Create a new model thread
        
        Function signature
        ------------------
        
        __init__ (n=0, block=10) 

        Function arguments
        ------------------        

        n -- Number of evaluations to perform

        block -- Number of consecutive evalutations to perform        

        """

        threading.Thread.__init__ (self)
        self.model = model
        self.n = n
        self.block = block
        self.stop = False

    def run (self):
        """

        Start thread
        
        Function signature
        ------------------
        
        run ()

        """

        if self.n:
            i = 0
            while (i < self.n) and not self.stop:
                if (i+self.block) < self.n:
                    self.model.evaluate (self.block)
                    i += self.block
                else:
                    self.model.evaluate (self.n-i)
                    i = self.n
                time.sleep (.0001)
        else:
            while not self.stop:
                self.model.evaluate (self.block)
                time.sleep (.0001)


# _________________________________________________________________________Model
class Model (_core.Model):
    """
    Threaded Model class
    """

    def __init__ (self, block = 1):
        """

        Create a new model
        
        Function signature
        ------------------
        
        __init__ (block=10) 

        Function arguments
        ------------------        
        
        block -- Number of consecutive evalutations to perform
        
        """

        _core.Model.__init__ (self)
        self.block = block


    def start (self, n=0):
        """

        Start model evaluation in a new thread

        Function signature
        ------------------
        
        start (n=0) 

        Function arguments
        ------------------        
        
        n -- Number of evaluations to perform

        """

        self.thread = ModelThread (self, n, self.block)
        if not self.thread.isAlive():
            self.thread.start()
        else:
            print "Model is already running"
        self.thread = None

        
    def stop (self):
        """

        Stop model evaluation

        Function signature
        ------------------
        
        stop () 

        """
        if hasattr (self, 'thread'):
            self.thread.stop = True
            self.thread.join()
