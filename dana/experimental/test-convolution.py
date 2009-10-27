

import dana
import numpy as np

S = dana.ones((50,50))
D = dana.zeros((50,50))
K = np.random.random((100,100))

D.connect(S, K, 'I', shared=False)
D.dV = 'I'
D.compute()
print D.V

D.connect(S, K, 'I', shared=True)
D.dV = 'I'
D.compute()
print D.V

D.connect(S, K, 'I', sparse=False)
D.dV = 'I'
D.compute()
print D.V

