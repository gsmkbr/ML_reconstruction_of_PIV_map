class StatEvaluation:
    def __init__(self, V_Orig, VReconst, SelectedRegressors, GappyProp, VelMapProp):
        self.VOrig = V_Orig.VOrigDuplicate
#        self.VReconstSVM = V_Reconst.V_predicted_SVR
#        self.VReconstMLP = V_Reconst.V_predicted_MLP
#        self.VReconstGPOD = V_POD.UPOD
        self.VReconst = {}
        for rgrs in SelectedRegressors:
            self.VReconst[rgrs] = VReconst[rgrs]
        self.LocationsForEvaluation = GappyProp.LocationsForEvaluation
        
    def ElementaryEvaluation(self, SelectedRegressors, UID, SnapIndex, EvaluatingComponent):
        import numpy as np
        VTemp = self.VOrig[SnapIndex,:,:,EvaluatingComponent]
        self.Target = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
        self.Output = {}
        for rgrs in SelectedRegressors:
            VTemp = np.array(self.VReconst[rgrs][SnapIndex,:,:,EvaluatingComponent])
            self.Output[rgrs] = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
#        VTemp = np.array(self.VReconstSVM[SnapIndex,:,:,EvaluatingComponent])
#        self.OutputSVM = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
#        VTemp = np.array(self.VReconstMLP[SnapIndex,:,:,EvaluatingComponent])
#        self.OutputMLP = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
#        VTemp = self.VReconstGPOD[SnapIndex,:,:,EvaluatingComponent]
#        self.OutputGPOD = VTemp[np.array(self.LocationsForEvaluation[SnapIndex,:,:,EvaluatingComponent])]
        
        # percent of deviation between prediction of models and the target values 
        # at test locations 
        self.PredDeviation = {}
        #Calculation of normalized absolute error
        if UID.PredicitonErrorType == 'Normalized':
            for rgrs in SelectedRegressors:
                self.PredDeviation[rgrs] = np.divide(np.fabs(self.Output[rgrs]-self.Target), np.fabs(self.Target))
#            self.SVMPredDeviation = np.divide(np.fabs(self.OutputSVM-self.Target), np.fabs(self.Target))
#            self.MLPPredDeviation = np.divide(np.fabs(self.OutputMLP-self.Target), np.fabs(self.Target))
#            self.GPODPredDeviation = np.divide(np.fabs(self.OutputGPOD-self.Target), np.fabs(self.Target))
        elif UID.PredicitonErrorType == 'Absolute':
            for rgrs in SelectedRegressors:
                self.PredDeviation[rgrs] = np.fabs(self.Output[rgrs]-self.Target)
#            self.SVMPredDeviation = np.fabs(self.OutputSVM-self.Target)
#            self.MLPPredDeviation = np.fabs(self.OutputMLP-self.Target)
#            self.GPODPredDeviation = np.fabs(self.OutputGPOD-self.Target)
        from sklearn.metrics import r2_score
        self.PredMNAE = {}
        self.PredRMSNAE = {}
        self.PredMedNAE = {}
        self.PredR2_Score = {}
        # Calculation of Mean Normalized Absolute Error (NMAE)
        for rgrs in SelectedRegressors:
            print(rgrs)
            self.PredMNAE[rgrs] = np.mean(self.PredDeviation[rgrs])
            self.PredRMSNAE[rgrs] = np.std(self.PredDeviation[rgrs])
            self.PredMedNAE[rgrs] = np.median(self.PredDeviation[rgrs])
            self.PredR2_Score[rgrs] = r2_score(self.Target, self.Output[rgrs])
#        self.SVMPredMNAE = np.mean(self.SVMPredDeviation)
#        self.MLPPredMNAE = np.mean(self.MLPPredDeviation)
#        self.GPODPredMNAE = np.mean(self.GPODPredDeviation)
        # Calculation of Root Mean Square of Normalized Absolute Error (NRMSE)
#        self.SVMPredRMSNAE = np.std(self.SVMPredDeviation)
#        self.MLPPredRMSNAE = np.std(self.MLPPredDeviation)
#        self.GPODPredRMSNAE = np.std(self.GPODPredDeviation)
        # Calculation of Median of Normalized Absolute Error (MedNAE)
#        self.SVMPredMedNAE = np.median(self.SVMPredDeviation)
#        self.MLPPredMedNAE = np.median(self.MLPPredDeviation)
#        self.GPODPredMedNAE = np.median(self.GPODPredDeviation)
        # Calculation of R2-score of Predictions
        
#        self.SVMPredR2_Score = r2_score(self.Target, self.OutputSVM)
#        self.MLPPredR2_Score = r2_score(self.Target, self.OutputMLP)
#        self.GPODPredR2_Score = r2_score(self.Target, self.OutputGPOD)
                
        
    def Results(self, SelectedRegressors, SnapIndex, component, ModelEvalWorkSheet, VelMapProp):
        print("_____________________________________________________________")
        for rgrs in SelectedRegressors:
            print("Mean Normalized Absolute Error-"+rgrs+": {}".format(self.PredMNAE[rgrs]))
#        print("Mean Normalized Absolute Error-GPOD: {}".format(self.GPODPredMNAE))
#        print("Mean Normalized Absolute Error-SVM: {}".format(self.SVMPredMNAE))
#        print("Mean Normalized Absolute Error-MLP: {}".format(self.MLPPredMNAE))
        
        print("_____________________________________________________________")
        for rgrs in SelectedRegressors:
            print("RMS of Normalized Absolute Error-"+rgrs+": {}".format(self.PredRMSNAE[rgrs]))
#        print("RMS of Normalized Absolute Error-GPOD: {}".format(self.GPODPredRMSNAE))
#        print("RMS of Normalized Absolute Error-SVM: {}".format(self.SVMPredRMSNAE))
#        print("RMS of Normalized Absolute Error-MLP: {}".format(self.MLPPredRMSNAE))
        
        print("_____________________________________________________________")
        for rgrs in SelectedRegressors:
            print("Median of Normalized Absolute Error-"+rgrs+": {}".format(self.PredMedNAE[rgrs]))
#        print("Median of Normalized Absolute Error-GPOD: {}".format(self.GPODPredMedNAE))
#        print("Median of Normalized Absolute Error-SVM: {}".format(self.SVMPredMedNAE))
#        print("Median of Normalized Absolute Error-MLP: {}".format(self.MLPPredMedNAE))
        
        print("_____________________________________________________________")
        for rgrs in SelectedRegressors:
            print("R2 Score-"+rgrs+": {}".format(self.PredR2_Score[rgrs]))
#        print("R2 Score-GPOD: {}".format(self.GPODPredR2_Score))
#        print("R2 Score-SVM: {}".format(self.SVMPredR2_Score))
#        print("R2 Score-MLP: {}".format(self.MLPPredR2_Score))
        
        # Exporting the results into the Excel file ('ModelsEvaluation' worksheet)
        import xlsxwriter
        if SnapIndex == VelMapProp.StartSnapIndex-1 and component==0:
            self.SnapRecord = 0
            KeyWords2 = ['Snapshot Index', 'Velocity Component', 'Predictive Model', 'MNAE', 'RMSNAE', 'MedNAE', 'R2 Score']
            Row=0
            for Col in range(0,7):
                ModelEvalWorkSheet.write(Row, Col, KeyWords2[Col])
        self.SnapRecord+=1
        KeyWords1 = ['SVR', 'MLP', 'GPOD']
        
        Statistics = [[self.PredMNAE['SVR'], self.PredRMSNAE['SVR'], self.PredMedNAE['SVR'], self.PredR2_Score['SVR']],
                      [self.PredMNAE['MLP'], self.PredRMSNAE['MLP'], self.PredMedNAE['MLP'], self.PredR2_Score['MLP']],
                      [self.PredMNAE['GPOD'], self.PredRMSNAE['GPOD'], self.PredMedNAE['GPOD'], self.PredR2_Score['GPOD']]]
        if component==0:
            Row = 1+(SnapIndex)*3*VelMapProp.NoComponents
            RefCol = 0
            for i in range(0,3*VelMapProp.NoComponents):
                ModelEvalWorkSheet.write(Row+i, RefCol, SnapIndex+1)
        
        Row = 1+(self.SnapRecord-1)*3
        RefCol = 1
        for i in range(0,VelMapProp.NoComponents):
            ModelEvalWorkSheet.write(Row+i, RefCol, component+1)
        
        for i in range(0,3):
            Row = 1+(self.SnapRecord-1)*3+i
            RefCol = 2
            ModelEvalWorkSheet.write(Row, RefCol, KeyWords1[i])
            for j in range (0,4):
                Col = RefCol+1+j
                ModelEvalWorkSheet.write(Row, Col, Statistics[i][j])
            if i==2 and component==VelMapProp.NoComponents-1 and SnapIndex == VelMapProp.EndSnapIndex-1:
                ModelEvalWorkSheet.write(Row+1, 2, 'Average')
                Labels = ['D', 'E', 'F', 'G']
                for j in range(0,4):
                    Formula = '=SUBTOTAL(1,' + Labels[j] + '2:' +Labels[j] + str(Row+1) + ')'
                    ModelEvalWorkSheet.write_formula(Row+1, 3+j, Formula)
                