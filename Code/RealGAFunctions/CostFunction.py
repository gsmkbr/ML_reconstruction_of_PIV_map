def CostFunction(OptParams, ModelParams):
    from math import floor
    UsedSamplesPercentage = 20
    nSamples = floor(UsedSamplesPercentage/100*OptParams['V_Reconst'].InputTrain.shape[0])
    nCV = 2
    nParallelJobs = 4
    if OptParams['OptKey'] == 'SVR':
        if OptParams['UID'].OptimizationScoringMethod == 'ClassicPrediction':
            from statistical_evaluation_svr import StatEvaluationSVR
            #Applying SVR on the original data in order to train the model
            OptParams['V_Reconst'].SVRTrain(OptParams['UID'], Penalty=ModelParams[0], Gamma = ModelParams[1], Epsilon=ModelParams[2])
            #Applying svr object to the input data to predict and reconstruct data
            OptParams['V_Reconst'].SVRReconstruct(OptParams['ActiveFeat'], OptParams['V_Orig'], OptParams['VelMapProp'], OptParams['SnapIndex'], OptParams['PredictedComponent'])
            StatResults = StatEvaluationSVR(OptParams['V_Orig'], OptParams['V_Reconst'], OptParams['GappyProp'], OptParams['VelMapProp'])
            StatResults.ElementaryEvaluation(OptParams['SnapIndex'], EvaluatingComponent = OptParams['PredictedComponent'], SVMOptCostMetric=OptParams['SVMOptCostMetric'])
            #StatResults.Results()
            Cost = StatResults.SVMOptCostValue
        elif OptParams['UID'].OptimizationScoringMethod == 'CrossValidation':
            from sklearn.svm import SVR
            svr = SVR(kernel = OptParams['UID'].SVRKernelType, C=ModelParams[0], gamma = ModelParams[1], epsilon=ModelParams[2])
            from sklearn.model_selection import cross_val_score
            CVS = cross_val_score(svr, OptParams['V_Reconst'].InputTrain.iloc[:nSamples, :], OptParams['V_Reconst'].TargetTrain[:nSamples], cv=nCV, scoring='r2', n_jobs=nParallelJobs)
            import numpy as np
            CVS = np.mean(np.array(CVS))
            Cost = 1 - CVS
        return Cost
    elif OptParams['OptKey'] == 'MLP':
        
        if ModelParams.size == 1:
            HLSize = (int(ModelParams[0]),)
        else:
            HLSize = []
            for i in range(0,ModelParams.size):
                HLSize.append(int(ModelParams[i]))
            HLSize = tuple(HLSize)
        
        if OptParams['UID'].OptimizationScoringMethod == 'ClassicPrediction':
            from statistical_evaluation_mlp import StatEvaluationMLP
            OptParams['V_Reconst'].MLPTrain(OptParams['UID'], HidLayerSize = HLSize, ActivFunc = OptParams['ActivFunc'])
            #Applying mlp object to the input data to predict and reconstruct data
            OptParams['V_Reconst'].MLPReconstruct(OptParams['ActiveFeat'], OptParams['V_Orig'], OptParams['VelMapProp'], OptParams['SnapIndex'], OptParams['PredictedComponent'])
            StatResults = StatEvaluationMLP(OptParams['V_Orig'], OptParams['V_Reconst'], OptParams['GappyProp'], OptParams['VelMapProp'])
            StatResults.ElementaryEvaluation(OptParams['SnapIndex'], EvaluatingComponent = OptParams['PredictedComponent'], MLPOptCostMetric=OptParams['MLPOptCostMetric'])
            #StatResults.Results()
            Cost = StatResults.MLPOptCostValue
        elif OptParams['UID'].OptimizationScoringMethod == 'CrossValidation':
            from sklearn.neural_network import MLPRegressor
            RandomState = 55
            mlp = MLPRegressor(hidden_layer_sizes = HLSize, activation = OptParams['ActivFunc'], early_stopping=OptParams['UID'].EarlyStopStatus, validation_fraction=OptParams['UID'].ValidFraction, tol=OptParams['UID'].ValidationTol, max_iter=OptParams['UID'].MLPMaxIter, solver=OptParams['UID'].Solver, learning_rate=OptParams['UID'].LearnRateStatus, learning_rate_init=OptParams['UID'].LearnRateInit, power_t=OptParams['UID'].PowerT, momentum=OptParams['UID'].MomentumCoeff, random_state=RandomState)
            from sklearn.model_selection import cross_val_score
            CVS = cross_val_score(mlp, OptParams['V_Reconst'].InputTrain.iloc[:nSamples, :], OptParams['V_Reconst'].TargetTrain[:nSamples], cv=nCV, scoring='r2', n_jobs=nParallelJobs)
            import numpy as np
            CVS = np.mean(np.array(CVS))
            Cost = 1 - CVS
        return Cost