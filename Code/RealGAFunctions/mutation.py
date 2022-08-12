def Mutate(p, mu, VarMin, VarMax):
    nVar = p.size
    import math
    n_mu = math.ceil(mu*nVar)
    import numpy as np
    RandPerm = np.random.permutation(np.arange(0, nVar))
    MutationPosition = RandPerm[0:n_mu]
    Sigma = 0.1 * (VarMax - VarMin).reshape(nVar,)
    
    y = p.copy()
    y[MutationPosition] = p[MutationPosition] + np.multiply(Sigma[MutationPosition],np.random.randn(n_mu,))
    y = np.maximum(y, VarMin)
    y = np.minimum(y, VarMax)
    return y