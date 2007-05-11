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


class LayerDefaultValue (unittest.TestCase):
    def setUp (self):
        self.layer = core.Layer()
        
    def testSpec (self):
        """ Check default spec is None """
        self.assertEqual (self.layer.spec, None)

    def testSize (self):
        """ Check layer is actually empty """
        self.assertEqual (len(self.layer), 0)


class LayerAccess (unittest.TestCase):
    def setUp (self):
        self.layer = core.Layer()

    def testAccessWhenEmpty_1 (self):
        """ Check we got IndexError when accessing an empty layer """
        self.assertRaises (IndexError, self.layer.__getitem__, 0)
        self.assertRaises (IndexError, self.layer.unit, 0)
        self.assertRaises (IndexError, self.layer.unit, 0, 0)

    def testAccess (self):
        """ Check we got IndexError when accessing beyond layer """
        m = core.Map ((1,2), (0,0))
        m.append (self.layer)
        self.layer.fill(core.Unit)
        self.assertRaises (IndexError, self.layer.__getitem__, 3)
        self.assertRaises (IndexError, self.layer.unit, 3)        
        self.assertRaises (IndexError, self.layer.unit, 1,0)
    

class LayerManagement (unittest.TestCase):
    def setUp (self):
        self.layer = core.Layer()

    def testFillNoShape (self):
        """ Check we cannot fill layer without a shape """
        self.assertRaises (AssertionError, self.layer.fill, core.Unit)

    def testFill (self):
        """ Check layer fill function """
        m = core.Map ((2,2), (0,0))
        m.append (self.layer)
        self.layer.fill(core.Unit)
        self.assertEqual (len(self.layer), 4)

    def testFillWithGarbage (self):
        """ Check layer fill function """
        m = core.Map ((2,2), (0,0))
        m.append (self.layer)
        self.assertRaises (TypeError, self.layer.fill, 1)

    def testIterable (self):
        """ Check layer is iterable """
        m = core.Map ((10,10), (0,0))
        m.append (self.layer)
        self.layer.fill(core.Unit)
        i = 0
        for u in self.layer:
            i = i+1
        self.assertEqual (i, 100)


class LayerAttributes (unittest.TestCase):
    def setUp (self):
        self.layer = core.Layer()

    def testPotentialsShape (self):
        """ Check potentials shape """
        m = core.Map ((2,4), (0,0))
        m.append (self.layer)
        self.layer.fill(core.Unit)
        potentials = self.layer.potentials()
        self.assertEqual (potentials.shape, (4,2))

    def testPotentials (self):
        """ Check potentials value """
        m = core.Map ((1,3), (0,0))
        m.append (self.layer)
        self.layer.fill(core.Unit)
        self.layer.unit(0,0).potential = 1.0
        self.layer.unit(0,1).potential = 2.0
        self.layer.unit(0,2).potential = 3.0
        potentials = self.layer.potentials()


class LayerFunctions (unittest.TestCase):
    def setUp (self):
        self.layer = core.Layer()
        
    def testComputeDP (self):
        """ Check potential computation """
        self.assertEqual (self.layer.compute_dp(), 0.0)

    def testComputeDW (self):
        """ Check weight computation """
        self.assertEqual (self.layer.compute_dw(), 0.0)


if __name__ == "__main__":
    unittest.main()
