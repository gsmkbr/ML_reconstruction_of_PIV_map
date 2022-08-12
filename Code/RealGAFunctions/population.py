class Population:
    def __init__(self, nPop, nVar):
        import numpy as np
        self.position = np.zeros((nPop, nVar))
        self.cost = np.zeros((nPop, 1))
        
    