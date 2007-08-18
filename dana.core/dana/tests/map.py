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
# $Id: map.py 161 2007-05-11 13:03:44Z rougier $

import unittest
import dana.core as core


class MapTests (unittest.TestCase):
    def setUp (self):
        self.map = core.Map()
        
    def testSpec (self):
        """ Check map default spec is None """
        self.assertEqual (self.map.spec, None)

    def testShape (self):
        """ Check map default shape is (0,0) """
        self.assertEqual (self.map.shape, (0,0) )

    def testPosition (self):
        """ Check map default position is (0,0)"""
        self.assertEqual (self.map.shape, (0,0) )

    def testFrame (self):
        """ Check map default frame is (0,0,1,1) """
        self.assertEqual (self.map.frame, (0,0,1,1) )

# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(MapTests)

if __name__ == "__main__":
    unittest.main()
