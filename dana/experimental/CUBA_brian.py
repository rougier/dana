import time
from brian import *

eqs = '''
    dv/dt = (ge+gi-(v+49*mV))/(20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
'''

n = 1

P = NeuronGroup(4000*n, eqs, threshold=-50*mV, reset=-60*mV)
P.v = -60*mV+10*mV*rand(len(P))
Pe = P.subgroup(3200*n)
Pi = P.subgroup(800*n)
Ce = Connection(Pe, P, 'ge', weight=1.62*mV, sparseness=0.02)
Ci = Connection(Pi, P, 'gi', weight=-9*mV, sparseness=0.02)

defaultclock.dt = .1*ms

M = SpikeMonitor(P)

t =time.clock()
run(0.5*second)
print time.clock()-t

raster_plot(M)
show()
