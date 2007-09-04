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
# $Id: unit.py 278 2007-08-17 06:23:47Z rougier $

import unittest
import dana.core as core


class SpecTests (unittest.TestCase):
    def setUp (self):
        self.model = core.Model()
        self.network = core.Network()
        self.map = core.Map ( (10,10) )
        self.layer = core.Layer()
        self.model.append (self.network)
        self.network.append (self.map)
        self.map.append (self.layer)
        self.layer.fill (core.Unit)
        
    
    def testInitialSpec (self):
        """ Check spec initial is None """
        self.assertEqual (self.model.spec, None)
        self.assertEqual (self.network.spec, None)
        self.assertEqual (self.map.spec, None)
        self.assertEqual (self.layer.spec, None)
        for unit in self.layer:
            self.assertEqual (unit.spec, None)

    def testSpecInheritance (self):
        """ Check spec inheritance """
        self.model.spec = core.Spec()
        self.assertEqual (self.network.spec, self.model.spec)
        self.assertEqual (self.map.spec, self.model.spec)
        self.assertEqual (self.layer.spec, self.model.spec)
        for unit in self.layer:
            self.assertEqual (unit.spec, self.model.spec)

    def testSpecInheritanceOverride (self):
        """ Check spec inheritance override """
        self.network.spec = core.Spec()
        self.map.spec = core.Spec()

        self.assertEqual (self.model.spec, None)
        self.assertNotEqual (self.network.spec, self.map.spec)
        self.assertEqual (self.layer.spec, self.map.spec)


# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(SpecTests)

if __name__ == "__main__":
    unittest.main()
