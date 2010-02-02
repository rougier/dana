import dana
import numpy as np


n       = 40
dt      = 0.1
alpha   = 12.5
tau     = 0.75
h       = 0.1
min_act = -1.0
max_act =  1.0

input = dana.group((n,n), name='input')
focus = dana.group((n,n), name='focus')

W = np.ones((1,1))*1.5
focus.connect(input, W, 'I', shared=True)

W = 3.15*dana.gaussian((2*n+1,2*n+1), 0.05)
    - 0.7*dana.gaussian((2*n+1,2*n+1), 0.1)
W[n,n] = 0
focus.connect (focus, W, 'L', shared=True)


focus.dV = 'minimum(maximum(V+dt/tau*(-V+(L+I+h)/alpha),min_act), max_act)'

input.V  = dana.gaussian((n,n), 0.2, ( 0.5, 0.5))
input.V += dana.gaussian((n,n), 0.2, (-0.5,-0.5))
input.V += (2*numpy.random.random((n,n))-1)*.05

n = 250
t = time.clock()
for i in range(n):
    focus.compute(dt)
print time.clock()-t
