def RealGA(OptParams):

    # Problem definition
    from RealGAFunctions.CostFunction import CostFunction
    import numpy as np
    nVar = OptParams['nVar']
    VarSize = (1,nVar)
    VarMin = OptParams['GAMinVar']
    VarMax = OptParams['GAMaxVar']
    
    
    # GA Parameters
    MaxIt = 100
    nPop = 20
    pc = 0.6
    nc = 2*round(pc*nPop/2)
    pm = 0.2
    nm = round(pm*nPop)
    mu = 0.01
    gamma = 0.0
    beta = 10.0
    
    UseRoulleteWheelSelection = False
    UseTournamentSelection = True
    UseRandomSelection = False
    
    if UseTournamentSelection:
        TournamentSize = 3
    
    # Initialization
    from RealGAFunctions.population import Population
    pop = Population(nPop, nVar)
    pop.position = np.random.uniform(np.repeat(VarMin, nPop, axis=0), np.repeat(VarMax, nPop, axis=0), pop.position.shape)
    for i in range (0, nPop):
        pop.cost[i,0] = CostFunction(OptParams, pop.position[i,:])
        if OptParams['OptKey'] == 'SVR':
            print('{0}, SnapIdx:{1:1d}, Comp:{2:1d}, Init. Stage, iter:{3:3d}, Opt. Param:{4}, Cost:{5:7.3f}'.format(OptParams['OptKey'], OptParams['SnapIndex'], OptParams['PredictedComponent'], i+1, pop.position[i,:], pop.cost[i,0]))
        elif OptParams['OptKey'] == 'MLP':
            print('{0}, SnapIdx:{1:1d}, Comp:{2:1d}, Init. Stage, iter:{3:3d}, Opt. Param:{4}, Cost:{5:7.3f}'.format(OptParams['OptKey'], OptParams['SnapIndex'], OptParams['PredictedComponent'], i+1, np.floor(pop.position[i,:]), pop.cost[i,0]))
    
    ArgSort = np.argsort(pop.cost, axis=0)
    pop.cost = np.sort(pop.cost, axis=0)
    pop.position = pop.position[ArgSort].reshape(nPop, nVar)
    
    pop.BestPosition = pop.position[0,:]
    pop.BestCost = np.zeros((MaxIt, 1))
    
    pop.WorstCost = pop.cost[-1,0]
    
    # Main Loop
    from RealGAFunctions.crossover import CrossOver
    from RealGAFunctions.mutation import Mutate
    from RealGAFunctions.selection_methods import RouletteWheelSelection, TournamentSelection
    import random
    for iteration in range (0, MaxIt):
        
        #Calculating selection probabilities
        P = np.exp(-beta*pop.cost/pop.WorstCost)
        P = P/np.sum(P)
        
        # Crossover
        popc_position = np.zeros((int(nc), nVar))
        popc_cost = np.zeros((int(nc), 1))
        for k in range (0, int(nc/2)):
            if UseRoulleteWheelSelection:
                i1 = RouletteWheelSelection(P)
                i2 = RouletteWheelSelection(P)
            elif UseTournamentSelection:
                i1 = TournamentSelection(pop, nPop, TournamentSize)
                i2 = TournamentSelection(pop, nPop, TournamentSize)
            elif UseRandomSelection:
                i1 = random.randint(0,nPop-1)
                i2 = random.randint(0,nPop-1)
            p1 = pop.position[i1,:]
            p2 = pop.position[i2,:]
            popc_position[2*k, :], popc_position[2*k+1, :] = CrossOver(p1, p2, gamma, VarMin, VarMax)
            popc_cost[2*k,0] = CostFunction(OptParams, popc_position[2*k,:])
            popc_cost[2*k+1,0] = CostFunction(OptParams, popc_position[2*k+1,:])
            
        # Mutation
        popm_position = np.zeros((int(nm), nVar))
        popm_cost = np.zeros((int(nm), 1))
        for k in range (0, int(nm)):
            i = random.randint(0,nPop-1)
            p = pop.position[i,:]
            popm_position[k,:] = Mutate(p, mu, VarMin, VarMax)
            popm_cost[k,0] = CostFunction(OptParams, popm_position[k,:])
            
        pop.position = np.concatenate((pop.position, popc_position, popm_position), axis=0)
        pop.cost = np.concatenate((pop.cost, popc_cost, popm_cost), axis=0)
        
        ArgSort = np.argsort(pop.cost, axis=0)
        pop.cost = np.sort(pop.cost, axis=0)
        pop.position = pop.position[ArgSort].reshape(pop.position.shape)
        
        pop.position = pop.position[0:nPop]
        pop.cost = pop.cost[0:nPop]
        
        pop.BestPosition = pop.position[0,:]
        pop.BestCost[iteration,0] = pop.cost[0,0]
        
        pop.WorstCost = max(pop.WorstCost, pop.cost[-1,0])
        
        if OptParams['OptKey'] == 'SVR':
            print('{0}, SnapIdx:{1:1d}, Comp:{2:1d}, Evolution. Stage, iter:{3:3d}, Opt. Param:{4}, Cost:{5:7.3f}'.format(OptParams['OptKey'], OptParams['SnapIndex'], OptParams['PredictedComponent'], iteration+1, pop.BestPosition, pop.BestCost[iteration,0]))
        elif OptParams['OptKey'] == 'MLP':
            print('{0}, SnapIdx:{1:1d}, Comp:{2:1d}, Evolution. Stage, iter:{3:3d}, Opt. Param:{4}, Cost:{5:7.3f}'.format(OptParams['OptKey'], OptParams['SnapIndex'], OptParams['PredictedComponent'], iteration+1, np.floor(pop.BestPosition), pop.BestCost[iteration,0]))
        
    
    import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(pop.BestCost)
    ax = OptParams['OptAxes']
    if OptParams['OptKey']=='SVR':
        Axes = ax[OptParams['PredictedComponent'],0]
    elif OptParams['OptKey']=='MLP':
        Axes = ax[OptParams['PredictedComponent'],1]
    AxesTitle = OptParams['OptKey']+ r' prediction of $U_{}^{}$'.format(OptParams['PredictedComponent']+1, OptParams['SnapIndex']+1)
    Axes.set_title(AxesTitle)
    Axes.plot(np.arange(1,MaxIt+1), pop.BestCost)
    Axes.set_xlabel('Iteration')
    Axes.set_ylabel('Cost Value')
    
    
    return pop.BestPosition