class DefiningMLModels:
    def __init__(self, V_Orig):
        self.NRow = V_Orig.NRow
        self.NCol = V_Orig.NCol
        
    def SVRTrain(self, UID, Penalty, Gamma, Epsilon):
        from sklearn.svm import SVR
        self.svr=SVR(kernel=UID.SVRKernelType,C=Penalty,gamma=Gamma, epsilon=Epsilon)
        self.svr.fit(self.InputTrain,self.TargetTrain)
        self.SVRTrainingScore=self.svr.score(self.InputTrain,self.TargetTrain)
        self.SVRPredictionScore=self.svr.score(self.InputTest,self.TargetTest)
        #print('SVR Reconstruction:\nKernel: {}\nGamma: {}\nC: {}\nTraining Score: {}\nPrediction Score: {}'.format(Kernel, str(Gamma), str(Penalty),str(SVRTrainingScore),str(SVRPredictionScore)))
    
    def SVR_CV_Score(self, UID, Penalty = 1.0, Gamma = 1.0, Epsilon=0.01):
        nCV = UID.CustomKFoldCVNo
        nParallelJobs = 4
        from sklearn.svm import SVR
        svr = SVR(kernel = UID.SVRKernelType, C=Penalty, gamma = Gamma, epsilon=Epsilon)
        from sklearn.model_selection import cross_val_score
        CVS = cross_val_score(svr, self.InputTrain, self.TargetTrain, cv=nCV, scoring='r2', n_jobs=nParallelJobs)
        import numpy as np
        self.SVR_CVScore_mean = np.mean(np.array(CVS))
        self.SVR_CVScore_StdDV = np.std(np.array(CVS))
    
    def SVR_CV_Train(self, UID, CVParams):
        from sklearn.svm import SVR
        #svr=SVR(kernel=UID.SVRKernelType)
        svr=SVR()
        if UID.CVStatus == 'GridSearchCV':
            from sklearn.model_selection import GridSearchCV
            self.svr = GridSearchCV(svr, CVParams, cv = UID.nCV, verbose=10)
        elif UID.CVStatus == 'RandomizedSearchCV':
            from sklearn.model_selection import RandomizedSearchCV
            self.svr = RandomizedSearchCV(svr, CVParams, cv = UID.nCV, n_iter=UID.RSCV_SVR_nIter, verbose=1, n_jobs=UID.No_Jobs_RSCV)
        self.svr.fit(self.InputTrain,self.TargetTrain)
        self.SVRTrainingScore=self.svr.score(self.InputTrain,self.TargetTrain)
        self.SVRPredictionScore=self.svr.score(self.InputTest,self.TargetTest)
        print('Best SVR Parameters for C, gamma, epsilon and kernel_type:{}'.format(self.svr.best_params_))
        
    def SVRReconstruct(self, ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = 'StandardScaler'):
        import pandas as pd
        import numpy as np
        V_predicted_SVR=[]
        NRows = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCols = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
            self.V_predicted_SVR = np.zeros((VelMapProp.NoSnapshots, self.NRow[str(VelMapProp.FOVIndex)], self.NCol[str(VelMapProp.FOVIndex)], VelMapProp.NoComponents))
            
        for i in range (0,NRows):
            for j in range (0,NCols):
                
                Position = [i+1,j+1]
                VGappyFiltered = []
                VorticityFiltered = []
                StrainFiltered = []
                for FiltSizeIndex in range (0,V_Orig.nFiltSizes):
                    VGappyFiltered.extend([V_Orig.VGappyFiltered[FiltSizeIndex,SnapIndex,i,j,PredictedComponent]])
                for FiltSizeIndex in range (0,V_Orig.nFiltSizes):
                    VorticityFiltered.extend([V_Orig.FilteredVorticity[FiltSizeIndex,SnapIndex,i,j]])
                for FiltSizeIndex in range (0,V_Orig.nFiltSizes):
                    for comp in range(0,3):
                        StrainFiltered.extend([V_Orig.FilteredStrain[FiltSizeIndex,SnapIndex,i,j,comp]])
                TargetQuantity = False
                ActiveFeat.SelectedFeatures(Position=Position, VGappyFiltered=VGappyFiltered, VorticityFiltered=VorticityFiltered, StrainFiltered=StrainFiltered, TargetQuantity=TargetQuantity)
                ##x_local=j+1
                ##y_local=i+1
                ##V_predicted_SVR.append(self.svr.predict(pd.DataFrame([[y_local,x_local,V_Orig.VGappyFiltered[0,SnapIndex,i,j,PredictedComponent],V_Orig.VGappyFiltered[1,SnapIndex,i,j,PredictedComponent]]])))
                ActivatedDataDF = pd.DataFrame([ActiveFeat.SelFeatures])
                if SCLMethod == 'None':
                    ActivatedDataDFScaled = ActivatedDataDF
                else:
                    ActivatedDataDFScaled = V_Orig.Scaler.transform(ActivatedDataDF)
                    ActivatedDataDFScaled = pd.DataFrame(ActivatedDataDFScaled)
                
                
                V_predicted_SVR.append(self.svr.predict(ActivatedDataDFScaled))
                
        V_predicted_SVR=np.array(V_predicted_SVR)
        V_predicted_SVR=V_predicted_SVR.reshape(NRows,NCols)
        self.V_predicted_SVR[SnapIndex,:,:,PredictedComponent] = pd.DataFrame(V_predicted_SVR)
        
    def MLPTrain(self, UID, HidLayerSize, ActivFunc):
        from sklearn.neural_network import MLPRegressor
        RandomState = 55
        self.mlp=MLPRegressor(hidden_layer_sizes = HidLayerSize, activation = ActivFunc, early_stopping=UID.EarlyStopStatus, validation_fraction=UID.ValidFraction, tol=UID.ValidationTol, max_iter=UID.MLPMaxIter, solver=UID.Solver, learning_rate=UID.LearnRateStatus, learning_rate_init=UID.LearnRateInit, power_t=UID.PowerT, momentum=UID.MomentumCoeff, random_state=RandomState)
        self.mlp.fit(self.InputTrain,self.TargetTrain)
        self.MLPTrainingScore=self.mlp.score(self.InputTrain,self.TargetTrain)
        self.MLPPredictionScore=self.mlp.score(self.InputTest,self.TargetTest)
        #print('MLP Reconstruction:\nHidden Layer Size: {}\nActivation Function: {}\nTraining Score: {}\nPrediction Score: {}'.format(str(HidLayerSize), ActivFunc, str(MLPTrainingScore),str(MLPPredictionScore)))
    
    def MLP_CV_Score(self, UID, HidLayerSize, ActivFunc = 'relu'):
        nCV = UID.CustomKFoldCVNo
        nParallelJobs = 4
        
        import numpy as np
        from sklearn.neural_network import MLPRegressor
        RandomState = 55
        mlp = MLPRegressor(hidden_layer_sizes = HidLayerSize, activation = ActivFunc, early_stopping=UID.EarlyStopStatus, validation_fraction=UID.ValidFraction, tol=UID.ValidationTol, max_iter=UID.MLPMaxIter, solver=UID.Solver, learning_rate=UID.LearnRateStatus, learning_rate_init=UID.LearnRateInit, power_t=UID.PowerT, momentum=UID.MomentumCoeff, random_state=RandomState)
        from sklearn.model_selection import cross_val_score
        CVS = cross_val_score(mlp, self.InputTrain, self.TargetTrain, cv=nCV, scoring='r2', n_jobs=nParallelJobs)
        
        self.MLP_CVScore_mean = np.mean(np.array(CVS))
        self.MLP_CVScore_StdDV = np.std(np.array(CVS))
    
    def MLP_CV_Train(self, UID, CVParams):
        from sklearn.neural_network import MLPRegressor
        RandomState = 55
        mlp=MLPRegressor(early_stopping=UID.EarlyStopStatus, validation_fraction=UID.ValidFraction, tol=UID.ValidationTol, max_iter=UID.MLPMaxIter, solver=UID.Solver, learning_rate=UID.LearnRateStatus, learning_rate_init=UID.LearnRateInit, power_t=UID.PowerT, momentum=UID.MomentumCoeff, random_state=RandomState)
        if UID.CVStatus == 'GridSearchCV':
            from sklearn.model_selection import GridSearchCV
            self.mlp = GridSearchCV(mlp, CVParams, cv = UID.nCV)
        elif UID.CVStatus == 'RandomizedSearchCV':
            from sklearn.model_selection import RandomizedSearchCV
            self.mlp = RandomizedSearchCV(mlp, CVParams, cv = UID.nCV, n_iter=UID.RSCV_MLP_nIter, verbose=1, n_jobs=UID.No_Jobs_RSCV)
            
        self.mlp.fit(self.InputTrain,self.TargetTrain)
        self.MLPTrainingScore=self.mlp.score(self.InputTrain,self.TargetTrain)
        self.MLPPredictionScore=self.mlp.score(self.InputTest,self.TargetTest)
        print('Best MLP Parameters for Layer size:{}'.format(self.mlp.best_params_))
    
    def MLPReconstruct(self, ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = 'StandardScaler'):
        import pandas as pd
        import numpy as np
        V_predicted_MLP=[]
        NRows = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCols = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
            self.V_predicted_MLP = np.zeros((VelMapProp.NoSnapshots, self.NRow[str(VelMapProp.FOVIndex)], self.NCol[str(VelMapProp.FOVIndex)], VelMapProp.NoComponents))
        for i in range (0,NRows):
            for j in range (0,NCols):
                
                Position = [i+1,j+1]
                VGappyFiltered = []
                VorticityFiltered = []
                StrainFiltered = []
                for FiltSizeIndex in range (0,V_Orig.nFiltSizes):
                    VGappyFiltered.extend([V_Orig.VGappyFiltered[FiltSizeIndex,SnapIndex,i,j,PredictedComponent]])
                for FiltSizeIndex in range (0,V_Orig.nFiltSizes):
                    VorticityFiltered.extend([V_Orig.FilteredVorticity[FiltSizeIndex,SnapIndex,i,j]])
                for FiltSizeIndex in range (0,V_Orig.nFiltSizes):
                    for comp in range(0,3):
                        StrainFiltered.extend([V_Orig.FilteredStrain[FiltSizeIndex,SnapIndex,i,j,comp]])
                TargetQuantity = False
                ActiveFeat.SelectedFeatures(Position=Position, VGappyFiltered=VGappyFiltered, VorticityFiltered=VorticityFiltered, StrainFiltered=StrainFiltered, TargetQuantity=TargetQuantity)
                ##x_local=j+1
                ##y_local=i+1
                ##V_predicted_MLP.append(self.mlp.predict(pd.DataFrame([[y_local,x_local,V_Orig.VGappyFiltered[0,SnapIndex,i,j,PredictedComponent],V_Orig.VGappyFiltered[1,SnapIndex,i,j,PredictedComponent]]])))
                ActivatedDataDF = pd.DataFrame([ActiveFeat.SelFeatures])
                if SCLMethod == 'None':
                    ActivatedDataDFScaled = ActivatedDataDF
                else:
                    ActivatedDataDFScaled = V_Orig.Scaler.transform(ActivatedDataDF)
                    ActivatedDataDFScaled = pd.DataFrame(ActivatedDataDFScaled)
                
                    
                V_predicted_MLP.append(self.mlp.predict(ActivatedDataDFScaled))
        V_predicted_MLP=np.array(V_predicted_MLP)
        V_predicted_MLP=V_predicted_MLP.reshape(NRows,NCols)
        self.V_predicted_MLP[SnapIndex,:,:,PredictedComponent] = pd.DataFrame(V_predicted_MLP)
        
    def VelocityMask(self, VelMapProp, UID):
        from math import sqrt
        NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        for i in range(0,NRow):
            for j in range(0,NCol):
                if UID.FOVIndex==1:
                    y_rotor = -143+sqrt(169**2-j**2)
                    if i<y_rotor:
                        self.V_predicted_SVR[:,i,j,:]=0
                        self.V_predicted_MLP[:,i,j,:]=0
                elif UID.FOVIndex==2:
                    y_rotor = -143+sqrt(169**2-(j-(NCol-1))**2)
                    if i<y_rotor:
                        self.V_predicted_SVR[:,i,j,:]=0
                        self.V_predicted_MLP[:,i,j,:]=0