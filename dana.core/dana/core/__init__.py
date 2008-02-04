#!/usr/bin/env python
#
#  DANA - Distributed Asynchronous Numerical Adaptive computing library
#  Copyright (c) 2006,2007,2008 Nicolas P. Rougier
#
#  This file is part of DANA.
#
#  DANA is free software: you can redistribute it and/or modify it under
#  the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your
#  option) any later version.
#
#  DANA is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with DANA. If not, see <http://www.gnu.org/licenses/>.

""" core

    The core package is a mandatory component of DANA that implements basic
    concepts of distributed computing. It offers a minimal set of classes that
    aims at representing the notion of units, layers, maps, networks and models
    that are necessary to run a simulation. The computational paradigm
    supporting the library is grounded on the notion of a unit that is seen as
    a potential that can vary along time under the influence of other units and
    learning. Those units are organized into layers, maps and network: a
    network is made of one to several map, a map is made of one to several
    layer and a layer is made of a set of several units. Each unit can be
    linked to any other unit (included itself) using a weighted link.
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
    """ Overloading of the write function of core.Object

        This function is able to access the current script in order to save a
        copy within the xml file.
    """
    co = sys._getframe(1).f_code
    script = "\n"
    f = file (co.co_filename, "r")
    for line in f:
        script += line
    f.close()
    if filename[-3:] != '.gz':
        filename += '.gz'
    self.__write (filename,  os.path.abspath (co.co_filename), script)
Object.write = __write


# ___________________________________________________________________ModelThread
class ModelThread (threading.Thread):
    """
    Model Thread class
    """

    def __init__ (self, model, n):
        """

        Create a new model thread
        
        Function signature
        ------------------
        
        __init__ (n=0) 

        Function arguments
        ------------------        

        n -- Number of evaluations to perform

        """

        threading.Thread.__init__ (self)
        self.model = model
        self.n = n
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
                self.model.evaluate (1)
                i += 1
        else:
            while not self.stop:
                self.model.evaluate (1) #self.block)


# _________________________________________________________________________Model
class Model (_core.Model):
    """
    Threaded Model class
    """

    def __init__ (self):
        """

        Create a new model
        
        Function signature
        ------------------
        
        __init__ () 
        
        """

        _core.Model.__init__ (self)
        self.thread = None


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

        if self.thread:
            if not self.thread.isAlive():
                self.thread.stop = True
                self.thread.join()
                self.thread = None
            else:
                print "Model is already running"
                return

        self.thread = ModelThread (self, n)
        self.thread.start()


        
    def stop (self):
        """

        Stop model evaluation

        Function signature
        ------------------
        
        stop () 

        """
        if self.thread:
            self.thread.stop = True
            self.thread.join()
            self.thread = None
