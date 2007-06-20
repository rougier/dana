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
# $Id$
#------------------------------------------------------------------------------

import matplotlib.pylab as pylab
import matplotlib.colors as colors

import dana.core as core
import dana.projection as proj
import dana.projection as projection
import dana.projection.distance as distance
import dana.projection.density as density
import dana.projection.shape as shape
import dana.projection.profile as profile

import dana.cnft as cnft
import dana.learn as learn

from glpython.window import window
from dana.visualization.gl.network import View
from dana.gui.gtk import ControlPanel

import time, random, math

import gobject, gtk

import numpy
import random
from threading import * 
from math import *

#   The purpose of this model is to classify even and odd numbers, using the 
#   perceptron algorithm 
#
#   The numbers are represented with 7 segments
#                  1
#              ---------
#             6|       | 2
#              |   7   |
#              ---------
#             5|       | 3
#              |       |
#              ---------
#                  4
#
#  Then, 0 is represented by (1111110)
#        1 is represented by (0110000)
#        2 is represented by (1101101)
#        3 is represented by (1111001)
#        4 is represented by (0010011)
#        5 is represented by (1011011)
#        6 is represented by (1011111)
#        7 is represented by (1110000)
#        8 is represented by (1111111)
#        9 is represented by (1111011)
#
#  To use it, simply call the function learn(nb_steps,lrate)
#  where nb_steps is the number of presentations of the whole set of digits
#        lrate    is the learning rate
#
#  To make some test, use clamp_ex(number) and run some steps
#  to see if the number is well classified

print "    Simulation of the perceptron, learning    "
print "to classify numbers in odd and even categories"
print " --   Open the file to know how to use it  -- "

numbers = []
# 0 is even
numbers.append([1,1,1,1,1,1,0])
numbers.append([1])
# 1 is odd
numbers.append([0,1,1,0,0,0,0])
numbers.append([0])
# 2 is even
numbers.append([1,1,0,1,1,0,1])
numbers.append([1])
# 3 is odd
numbers.append([1,1,1,1,0,0,1])
numbers.append([0])
# 4 is even
numbers.append([0,0,1,0,0,1,1])
numbers.append([1])
# 5 is odd
numbers.append([1,0,1,1,0,1,1])
numbers.append([0])
# 6 is even
numbers.append([1,0,1,1,1,1,1])
numbers.append([1])
# 7 is odd
numbers.append([1,1,1,0,0,0,0])
numbers.append([0])
# 8 is even
numbers.append([1,1,1,1,1,1,1])
numbers.append([1])
# 9 is odd
numbers.append([1,1,1,1,0,1,1])
numbers.append([0])

# Create a new network
model = core.Model()
net = core.Network ()
model.append(net)

# Create the map representing the 7 segments of a number
number = core.Map ( (1,7), (0,0) )
number.append(core.Layer())
number[0].fill(core.Unit)
number.name = 'number'
net.append(number)

# Create the map representing the result : even for unit(0), odd for unit(1)
evenodd = core.Map ( (1,2), (1,0) )
evenodd.append (core.Layer())
evenodd[0].fill(learn.Unit)
evenodd.name = 'evenodd'

evenodd.spec = cnft.Spec()
evenodd.spec.tau      = 0.75
evenodd.spec.baseline = 0.0
evenodd.spec.alpha    = 1.0
evenodd.spec.min_act  = 0.0
evenodd.spec.max_act  = 1.0

net.append(evenodd)

proj          = projection.projection()
proj.self     = True
proj.distance = distance.euclidean(False)
proj.density  = density.full(1)
proj.shape    = shape.box(1,1)
proj.profile  = profile.uniform(0,0)
proj.src      = number[0]
proj.dst      = evenodd[0]
proj.connect()

learner = learn.Learner()

################### Learning Rule ##########################
# Hebb's rule : dw/dt = lrate * vi * vj
#learner.set_source(number[0])
#learner.set_destination(evenodd[0])
#learner.add_one([1,1,[1.0]])
#learner.connect()

# Oja's rule : dw/dt = lrate*(vi*vj - wij * (vi**2))
#                    = lrate*vi*vj - lrate*wij*(vi**2)
learner.set_source(number[0])
learner.set_destination(evenodd[0])
learner.add_one([1,1,[1]])
learner.add_one([2,0,[0,-1]])
learner.connect()

# Custom rule : dw/dt = lrate * (vi - wij) * (vj - wij)
#                     = lrate * (wij**2) - lrate*vi*wij - lrate*vj*wij + lrate*vi*vj
#learner.set_source(number[0])
#learner.set_destination(evenodd[0])
#learner.add_one([0,0,[0,0,1.0]])
#learner.add_one([1,0,[0,-1.0]])
#learner.add_one([0,1,[0,-1.0]])
#learner.add_one([1,1,[1.0]])
#learner.connect()
############################################################

## Show network
current_step = 0
steps = []
error = []

def main_learn(nb_steps,lrate):
    t = Thread(target=learn, args=(nb_steps,lrate,1))
    gobject.idle_add(t.start)
        
def learn(nb_steps,lrate):
	global current_step,steps,error
	for n in range(nb_steps):
		i=0
		for i in range((len(numbers))/2):
			# Set the input
			clamp_ex(i)
			# Make some steps
			net.evaluate(4,False)
			# Update the output value to take into account the desired output
			clamp_res(i)
			# Learn
			learner.learn(lrate)
		# Make a test phase, and record the error
		steps.append(current_step)
		error.append(compute_error())
		current_step += 1

def clamp_ex(i):
	num = numbers[2*i]
	for j in range(len(num)):
		number[0].unit(j).potential = num[j]

def clamp_res(i):
	res = numbers[2*i+1]
	evenodd.unit(0).potential = res[0] - evenodd.unit(0).potential
	evenodd.unit(1).potential = (1-res[0]) - evenodd.unit(1).potential

def test(i):
	# Clamp the representation of number i, make some steps, and get the result "odd or even"
	clamp_ex(i)
	net.evaluate(4,False)
	if (evenodd.unit(0).potential > evenodd.unit(1).potential):
		print "Number ",i,"is even"
	else:
		print "Number ",i,"is odd"

def compute_error():
	err = 0.0
	for i in range(len(numbers)/2):
		clamp_ex(i)
		res = numbers[2*i + 1]
		net.evaluate(4,False)
		err += sqrt(pow(res[0] - evenodd.unit(0).potential,2.0) + pow((1-res[0]) - evenodd.unit(1).potential,2.0))
	return err

# Show network
win = window(locals(), backend='gtk')
win.view.append (View (net, fontsize=48))
control = ControlPanel (model)
win.show()

