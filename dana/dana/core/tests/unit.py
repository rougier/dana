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
        """ Check default potential is 0.0 """
        self.assertEqual (self.unit.potential, 0.0)

    def testSpec (self):
        """ Check default spec is None """
        self.assertEqual (self.unit.spec, None)

    def testPosition (self):
        """ Check default unit position is (-1,-1) """
        self.assertEqual (self.unit.position, (-1,-1))


class UnitAttributes(unittest.TestCase):
    def setUp (self):
        self.unit = core.Unit()

    def testSetPosition(self):
        """ Check position is read-only """
        self.assertRaises (AttributeError, setattr, self.unit,
                           'position', (0,0))
    def testWeightsNoArg(self):
        """ Check weights cannot be get without argument """
        self.assertRaises (TypeError, self.unit.weights)

    def testWeightsNoneArg (self):
        """ Check weights cannot be get with None argument """
        self.assertRaises (AssertionError, self.unit.weights, None)

    def testWeightsGarbage (self):
        """ Check weights cannot be get with argument of any type """
        self.assertRaises (TypeError, self.unit.weights, "ee")


class UnitFunctions (unittest.TestCase):
    def setUp (self):
        self.unit = core.Unit()
        
    def testComputeDP (self):
        """ Check potential computation """
        self.assertEqual (self.unit.compute_dp(), 0.0)

    def testComputeDW (self):
        """ Check weight computation """
        self.assertEqual (self.unit.compute_dw(), 0.0)

    def testConnect (self):
        """ Check connection """    
        other = core.Unit()
        self.unit.connect (other, 1.0)


if __name__ == "__main__":
    unittest.main()
