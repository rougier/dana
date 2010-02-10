import time
import numpy as np
import dana

class array(np.ndarray): pass

n = 50
epochs = 100000

# Numpy regular array
Z = np.zeros((n,n), dtype=np.double)
t = time.clock()
for i in range(epochs):
    Z += Z**2+0.1
print 'numpy:', time.clock()-t

# Numpy regular array + dynamic evaluation
t = time.clock()
for i in range(epochs):
    Z += eval("Z**2+0.1")
print 'numpy + dynamic eval:', time.clock()-t

# Numpy regular array + static evaluation
expr = compile("Z**2+0.1", "<string>", "eval")
t = time.clock()
for i in range(epochs):
    Z += eval(expr)
print 'numpy + static eval:', time.clock()-t

# Numpy regular array + dynamic execution
t = time.clock()
for i in range(epochs):
    exec('Z += Z**2+0.1')
print 'numpy + dynamic exec:', time.clock()-t

# Numpy regular array + static execution
expr = compile("Z += Z**2+0.1", "<string>", "exec")
t = time.clock()
for i in range(epochs):
    exec(expr)
print 'numpy + static exec:', time.clock()-t

# Numpy aligned interleaved array
Z = np.zeros((n,n),dtype=[('x',np.double), ('y',np.int)])['x']
t = time.clock()
for i in range(epochs):
    Z += Z**2+0.1
print 'aligned interleaved array:', time.clock()-t

# Numpy unaligned interleaved array
Z = np.zeros((n,n),dtype=[('x',np.double), ('y',np.bool)])['x']
t = time.clock()
for i in range(epochs):
    Z += Z**2+0.1
print 'unaligned interleaved array:', time.clock()-t

# Numpy subclass array
Z = array((n,n), dtype=np.double)
Z[...] = 0
t = time.clock()
for i in range(epochs):
    Z += Z**2+0.1
print 'subclass array:', time.clock()-t

# Numpy aligned interleaved subclass array
Z = array((n,n),dtype=[('x',np.double), ('y',np.int)])['x']
Z[...] = 0
t = time.clock()
for i in range(epochs):
    Z += Z**2+0.1
print 'aligned interleaved subclass array:', time.clock()-t

# Numpy unaligned interleaved subclass array
Z = array((n,n),dtype=[('x',np.double), ('y',np.bool)])['x']
Z[...] = 0
t = time.clock()
for i in range(epochs):
    Z += Z**2+0.1
print 'unaligned interleaved subclass array:', time.clock()-t

# Numpy unaligned interleaved subclass array
Z = dana.group((n,n), dtype=[('x',np.double), ('y',np.bool)])
Z.dx = 'x**2+0.1'
t = time.clock()
Z.compile()
for i in range(epochs):
    Z.compute()
print 'group:', time.clock()-t

