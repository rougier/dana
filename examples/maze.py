#!/usr/bin/env python
# -----------------------------------------------------------------------------
# DANA - Distributed (Asynchronous) Numerical Adaptive computing framework
# Copyright (C) 2009-2010  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License. The full license is in
# the file COPYING, distributed as part of this software.
# -----------------------------------------------------------------------------
'''
Maze solving using the Bellman-Ford algorithm
'''
from dana import *


def maze(shape=(64,64), complexity=.75, density = 1):
    # Only odd shapes
    shape = ((shape[0]//2)*2+1, (shape[1]//2)*2+1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity*(5*(shape[0]+shape[1])))
    density    = int(density*(shape[0]//2*shape[1]//2))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0,:] = Z[-1,:] = 1
    Z[:,0] = Z[:,-1] = 1
    # Make isles
    for i in range(density):
        x = shape[1]*(.5-min(max(np.random.normal(0,.5),-.5),.5))
        y = shape[0]*(.5-min(max(np.random.normal(0,.5),-.5),.5))
        x, y = (x//2)*2, (y//2)*2
        #x, y = rnd(0,shape[1]//2)*2, rnd(0,shape[0]//2)*2
        Z[y,x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:           neighbours.append( (y,x-2) )
            if x < shape[1]-2:  neighbours.append( (y,x+2) )
            if y > 1:           neighbours.append( (y-2,x) )
            if y < shape[0]-2:  neighbours.append( (y+2,x) )
            if len(neighbours):
                y_,x_ = neighbours[rnd.random_integers(0,len(neighbours)-1)]
                if Z[y_,x_] == 0:
                    Z[y_,x_] = 1
                    Z[y_+(y-y_)//2, x_+(x-x_)//2] = 1
                    x, y = x_, y_
    Z[ 0, 1] = 0
    Z[-2,-1] = 0
    return Z


n = 41
a = 0.99
Z = 1-maze((n,n))
G = Group((n,n),'''V = I*maximum(maximum(maximum(maximum(V,E),W),N),S)
                     W; E; N; S; I''')
SparseConnection(Z,   G('I'), np.array([ [1] ]))
SparseConnection(G.V, G('N'), np.array([ [a],      [np.NaN], [np.NaN] ]))
SparseConnection(G.V, G('S'), np.array([ [np.NaN], [np.NaN], [a]      ]))
SparseConnection(G.V, G('E'), np.array([ [np.NaN,  np.NaN,  a]        ]))
SparseConnection(G.V, G('W'), np.array([ [a,       np.NaN,  np.NaN]   ]))
G.V[-2,-1] = 1
run(n=5*(n+n))

# Visualization
# from ascii import *
# imshow(G.V,show_cmap=False)
plt.figure(figsize=(9,9))
cmap = plt.cm.hot
cmap.set_under('white')
cmap.set_over('black')
cmap.set_bad('blue')
plt.imshow(G.V, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
plt.xticks([]), plt.yticks([])

# Solution finding and drawing by ascending gradient
y,x = 0, 1
X,Y = [],[]
dirs = [(0,-1), (0,+1), (-1,0), (+1,0)]
S = Z.astype(float)
for i in range(5*(n+n)):
    Y.append(y), X.append(x)
    neighbours = -np.ones(4)
    if x > 0:            neighbours[0] = G.V[y,x-1]
    if x < G.shape[1]-1: neighbours[1] = G.V[y,x+1]
    if y > 0:            neighbours[2] = G.V[y-1,x]
    if y < G.shape[0]-1: neighbours[3] = G.V[y+1,x]
    a = np.argmax(neighbours)
    x,y  = x + dirs[a][1], y + dirs[a][0]

plt.scatter(X, Y, s=40.0, lw=2, color='k', marker='o',
            alpha=1.0, edgecolors='k', facecolors='w')
plt.axis( [-0.5,G.shape[1]-0.5, -0.5, G.shape[0]-0.5] )
plt.title('Maze path finding using\n Bellman-Ford algorithm', fontsize=20)
plt.show()
