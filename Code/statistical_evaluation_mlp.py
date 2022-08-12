class StatEvaluationMLP:
    def __init__(self, V_Orig, V_Reconst, GappyProp, VelMapProp):
        self.VOrig = V_Orig.VOrigDuplicate
        self.VReconstMLP = V_Reconst.V_predicted_MLP
        self.LocationsForEvaluation = GappyProp.LocationsForEvaluation
        
    def ElementaryEvaluation(self, SnapIndex, EvaluatingComponent, MLPOptCostMetric='R2_Score'):
        import numpy as np
        VTemp = self.VOrig[SnapIndex,:,:,EvaluatingComponent]
        self.Target = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
        VTemp = np.array(self.VReconstMLP[SnapIndex,:,:,EvaluatingComponent])
        self.OutputMLP = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
        
        # percent of deviation between prediction of models and the target values 
        # at test locations 
        self.MLPPredDeviation = np.divide(np.fabs(self.OutputMLP-self.Target), np.fabs(self.Target))
        self.MLPPredStdv = np.std(self.MLPPredDeviation)
        
        if MLPOptCostMetric == 'MNAE':
            # Calculation of Mean Normalized Absolute Error (NMAE)
            self.MLPOptCostValue = np.mean(self.MLPPredDeviation)
        elif MLPOptCostMetric == 'RMSNAE':
            # Calculation of Root Mean Square of Normalized Absolute Error (NRMSE)
            self.MLPOptCostValue = np.std(self.MLPPredDeviation)
        elif MLPOptCostMetric == 'MedNAE':
            # Calculation of Median of Normalized Absolute Error (MedNAE)
            self.MLPOptCostValue = np.median(self.MLPPredDeviation)
        elif MLPOptCostMetric == 'R2_Score':
            # Calculation of R2-score of Predictions
            from sklearn.metrics import r2_score
            self.MLPOptCostValue = 1-r2_score(self.Target, self.OutputMLP)
        
    def Results(self):
        print("StdDV of error-MLP: {}".format(self.MLPPredStdv))
        