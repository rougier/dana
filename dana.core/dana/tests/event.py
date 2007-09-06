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


class EventTests (unittest.TestCase):
    def setUp (self):
        self.model = core.Model()
        self.network = core.Network()
        self.map = core.Map ( (10,10) )
        self.layer = core.Layer()
        self.model.append (self.network)
        self.network.append (self.map)
        self.map.append (self.layer)
        self.layer.fill (core.Unit)
        
   
    def testEventDP (self):
        """ EventDP triggered by layer """
        class Observer(core.Observer):
            count = 0
            def notify (self, event):
                self.count += 1
        obs = Observer()
        core.EventDP.attach (obs)
        self.layer.compute_dp()
        self.assertEqual (obs.count, 100)

    def testEventDW (self):
        """ EventDW triggered by layer """
        class Observer(core.Observer):
            count = 0
            def notify (self, event):
                self.count += 1
        obs = Observer()
        core.EventDW.attach (obs)
        self.layer.compute_dw()
        self.assertEqual (obs.count, 100)


# Test suite
suite = unittest.TestLoader().loadTestsFromTestCase(EventTests)

if __name__ == "__main__":
    unittest.main()
