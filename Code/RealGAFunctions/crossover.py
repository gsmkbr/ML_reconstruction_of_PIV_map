def CrossOver(p1,p2, gamma, VarMin, VarMax):
    import numpy as np
    alpha = np.random.uniform(-gamma,1+gamma,p1.shape)
    y1 = np.multiply(alpha,p1)+np.multiply((1-alpha), p2)
    y2 = np.multiply(alpha,p2)+np.multiply((1-alpha), p1)    
    y1 = np.maximum(y1, VarMin)
    y1 = np.minimum(y1, VarMax)
    y2 = np.maximum(y2, VarMin)
    y2 = np.minimum(y2, VarMax)
    
    return y1, y2
    