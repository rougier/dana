import numpy, dana

Z = dana.group([[0,0,1,0],
                [1,0,1,0],
                [0,1,1,0],
                [0,0,0,0]])
Z.connect(Z.V, numpy.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]]), 'N')
Z.dV = '-V+maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'

print 'Initial state:'
print Z.V
for i in range(4):
    Z.compute()
print 'Final state:'
print Z.V
