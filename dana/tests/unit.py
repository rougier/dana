#!/usr/bin/env python
#
# Copyright (c) 2006-2007 Nicolas Rougier.
# All rights reserved.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the
# License, or (at your option) any later version.
# 
# $Id$

import unittest
import dana.core as core

class UnitDefaultValue (unittest.TestCase):
    def setUp (self):
        self.unit = core.Unit()
        
    def testPotential (self):
        """ Check default unit potential is 0.0 """
        self.assertEqual (self.unit.potential, 0.0,
                          'Default unit potential should be 0.0')

    def testSpec (self):
        """ Check default unit spec is None """
        self.assertEqual (self.unit.spec, None,
                          'Default unit specifications should be None')

    def testPosition (self):
        """ Check default unit position is (-1,-1) """
        self.assertEqual (self.unit.position, (-1,-1),
                          'Default unit position should be (-1,-1)')

class UnitPotentials (unittest.TestCase):
    def setUp (self):
        self.unit = core.Unit()
        
    def testComputeDP (self):
        """ Check potential computation """
        self.assertEqual (self.unit.compute_dp(), 0.0)

    def testComputeDW (self):
        """ Check potential computation """
        self.assertEqual (self.unit.compute_dw(), 0.0)

    def testConnect (self):
        """ Check connection """    
        other = core.Unit()
        self.unit.connect (other, 1.0)

if __name__ == "__main__":
    unittest.main()
