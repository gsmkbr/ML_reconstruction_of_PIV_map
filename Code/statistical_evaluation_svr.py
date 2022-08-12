class StatEvaluationSVR:
    def __init__(self, V_Orig, V_Reconst, GappyProp, VelMapProp):
        self.VOrig = V_Orig.VOrigDuplicate
        self.VReconstSVM = V_Reconst.V_predicted_SVR
        self.LocationsForEvaluation = GappyProp.LocationsForEvaluation
        
    def ElementaryEvaluation(self, SnapIndex, EvaluatingComponent, SVMOptCostMetric='R2_Score'):
        import numpy as np
        VTemp = self.VOrig[SnapIndex,:,:,EvaluatingComponent]
        self.Target = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
        VTemp = np.array(self.VReconstSVM[SnapIndex,:,:,EvaluatingComponent])
        self.OutputSVM = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
        
        
        # percent of deviation between prediction of models and the target values 
        # at test locations 
        self.SVMPredDeviation = np.divide(np.fabs(self.OutputSVM-self.Target), np.fabs(self.Target))
        self.SVMPredStdv = np.std(self.SVMPredDeviation)
        
        if SVMOptCostMetric == 'MNAE':
            # Calculation of Mean Normalized Absolute Error (NMAE)
            self.SVMOptCostValue = np.mean(self.SVMPredDeviation)
        elif SVMOptCostMetric == 'RMSNAE':
            # Calculation of Root Mean Square of Normalized Absolute Error (NRMSE)
            self.SVMOptCostValue = np.std(self.SVMPredDeviation)
        elif SVMOptCostMetric == 'MedNAE':
            # Calculation of Median of Normalized Absolute Error (MedNAE)
            self.SVMOptCostValue = np.median(self.SVMPredDeviation)
        elif SVMOptCostMetric == 'R2_Score':
            # Calculation of R2-score of Predictions
            from sklearn.metrics import r2_score
            self.SVMOptCostValue = 1-r2_score(self.Target, self.OutputSVM)
        
        
    def Results(self):
        print("StdDV of error-SVM: {}".format(self.SVMPredStdv))
        