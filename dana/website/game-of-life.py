import numpy, dana
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 50
G = dana.group((n//2,n), dtype=int)
K = numpy.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])
G.connect(G.V, K, 'N', sparse=True)
G.dV = '-V+maximum(0,1.0-(N<1.5)-(N>3.5)-(N<2.5)*(1-V))'
G.V = numpy.random.randint(0,2,G.shape)

for i in range(50):
    G.compute()

fig = plt.figure(figsize=(12,6))
fig.patch.set_alpha(0.0)

plt.imshow(G.V, cmap=plt.cm.gray_r, extent=[0,n,0,n//2],
           interpolation='nearest')
plt.yticks(numpy.arange(5)*5)
plt.xticks(numpy.arange(10)*5)
plt.grid()
fig.savefig('game-of-life.png', dpi=50)
#plt.show()
