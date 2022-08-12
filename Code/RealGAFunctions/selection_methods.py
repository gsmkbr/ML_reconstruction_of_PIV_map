def RouletteWheelSelection(P):
    import random
    import numpy as np
    r = random.random()
    c = np.cumsum(P)
    W = np.where(r<=c)
    index = W[0][0]
    return index

def TournamentSelection(pop, nPop, N_ts):
    import numpy as np
    Range = np.arange(0,nPop)
    Perm = np.random.permutation(Range)
    Perm = Perm[0:N_ts]
    SelectedCost = pop.cost[Perm,0]
    J = np.argmin(SelectedCost)
    index = Perm[J]
    return index