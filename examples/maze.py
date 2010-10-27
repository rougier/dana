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

def maze(shape=(64,64), complexity=.75, density =.5):
    assert len(shape) == 2
    assert shape[0] % 2 == shape[1] % 2 == 1

    complexity = int(complexity*(shape[0]+shape[1]))
    density = int(density*(shape[0]*shape[1])/4)
    shape = ((shape[0]//2)*2+1, (shape[1]//2)*2+1)
    Z = np.zeros(shape, dtype=int)
    Z[0,:] = Z[-1,:] = 1
    Z[:,0] = Z[:,-1] = 1
    for i in range(density):
        x = (rnd.random_integers(0,shape[1]//2))*2
        y = (rnd.random_integers(0,shape[0]//2))*2
        Z[x,y] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:          neighbours.append( (x-2,y) )
            if x < shape[1]-2: neighbours.append( (x+2,y) )
            if y > 1:          neighbours.append( (x,y-2) )
            if y < shape[0]-2: neighbours.append( (x,y+2) )
            if len(neighbours) == 0:
                continue
            d = rnd.random_integers(0,len(neighbours)-1)
            x_,y_ = neighbours[d]
            if Z[x_,y_] == 0:
                Z[x_,y_] = 1
                Z[x+(x_-x)//2, y+(y_-y)//2] = 1
                x, y = x_, y_
    Z[ 0, 1] = 0
    Z[-2,-1] = 0
    return Z


n = 43
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


# Path finding by ascending gradient
y,x = 0, 1
dirs = [(0,-1), (0,+1), (-1,0), (+1,0)]
S = Z.astype(float)
for i in range(5*(n+n)):
    S[y,x] = np.NaN
    g = np.ones(4)*-1
    if x > 0:            g[0] = G.V[y,x-1]
    if x < G.shape[1]-1: g[1] = G.V[y,x+1]
    if y > 0:            g[2] = G.V[y-1,x]
    if y < G.shape[0]-1: g[3] = G.V[y+1,x]
    a = np.argmax(g)
    x,y  = x + dirs[a][1], y + dirs[a][0]


# Visualization
plt.figure(figsize=(15,7))
cmap = plt.cm.hot
cmap.set_under('white')
cmap.set_over('black')
cmap.set_bad('blue')
plt.subplot(1,2,1)
plt.imshow(G.V, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(S, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
plt.xticks([]), plt.yticks([])
plt.show()
