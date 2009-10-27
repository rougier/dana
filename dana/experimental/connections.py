
import dana, dana.pylab
import numpy as np
import scipy.sparse as sp


n,m = 4,4
S = dana.zeros((n,m))
D = dana.zeros((n,m))

src_size = S.V.size
src_shape = S.V.shape
dst_size = S.V.size
dst_shape = S.V.shape

K = np.array([[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
D.connect(S.V, K, 'I')

# K = sp.lil_matrix((dst_size, src_size), dtype=np.float32)
# #K = np.zeros((dst_size, src_size), dtype=np.float32)
# scale = src_size/float(dst_size)
# for i in range(K.shape[0]):
#     #for j in range(D.V.shape[1]):
#     #index = i*D.V.shape[1]+j
#     #print D.V[i,j] - D.V.flatten()[index]
#     index = np.array(list(np.unravel_index(i, dst_shape)))
#     #print index
#     k = dana.extract(kernel, src_shape, (index*scale).astype(int), 0).flatten()
#     #print k.nonzero()
#     K[i,:] = k
#     #kernel = sp.csr_matrix(K)



view = dana.pylab.view([S.V, D.V])
view.show()
