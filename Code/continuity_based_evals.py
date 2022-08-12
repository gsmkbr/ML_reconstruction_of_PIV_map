class ContinuityBasedEvaluation:
    def __init__(self, VOrig, VReconst, SelectedRegressors, LocationsForVelEvaluation, VelMapProp, FOVIndex, SnapIndex):
        self.VOrig = VOrig
#        self.VPredSVR = V_predicted_SVR
#        self.VPredMLP = V_predicted_MLP
#        self.UPOD = UPOD
        self.VPred = {}
        for rgrs in SelectedRegressors:
            self.VPred[rgrs] = VReconst[rgrs][SnapIndex,:,:,:]
        self.LocForVelEVal = LocationsForVelEvaluation
        self.NRow = VelMapProp.NRow[str(FOVIndex-1)]
        self.NCol = VelMapProp.NCol[str(FOVIndex-1)]
        self.SnapIndex = SnapIndex
        
    def LocationsForContinuityEvals(self):
        self.ContEvalsLoc = []
#        for i in range(1,self.NRow-1):
#            for j in range(1,self.NCol-1):
        for i in range(30,self.NRow-5):
            for j in range(5,self.NCol-5):
                Cond1 = self.LocForVelEVal[i,j] == True
                Cond2 = self.VOrig[i+1,j,0]!=0
                Cond3 = self.VOrig[i-1,j,0]!=0
                Cond4 = self.VOrig[i,j+1,1]!=0
                Cond5 = self.VOrig[i,j-1,1]!=0
                if Cond1 and Cond2 and Cond3 and Cond4 and Cond5:
                #if Cond2 and Cond3 and Cond4 and Cond5:
                    self.ContEvalsLoc.append((i,j))
                    
    def ContinuityMetrics(self, SnapIndex, SelectedRegressors, XSpacing, YSpacing, UID):
#        self.ContMetricOrig = []
#        self.ContMetricPredSVR = []
#        self.ContMetricPredMLP = []
#        self.ContMetricPredGPOD = []
        self.ContMetric = {}
        self.ContMetric['Orig']=[]
        for rgrs in SelectedRegressors:
            self.ContMetric[rgrs] = []
        for sample in self.ContEvalsLoc:
            i = sample[0]
            j = sample[1]
            #Calculation of continuity metric (-S_xx-S_yy) for the original field at the evaluation locations
            strain_xx_Orig = (self.VOrig[i+1,j,0]-self.VOrig[i-1,j,0])/(2*XSpacing)
            strain_yy_Orig = (self.VOrig[i,j+1,1]-self.VOrig[i,j-1,1])/(2*YSpacing)
            self.ContMetric['Orig'].extend([-(strain_xx_Orig + strain_yy_Orig)])
            
            for rgrs in SelectedRegressors:
                strain_xx = (self.VPred[rgrs][i+1,j,0]-self.VPred[rgrs][i-1,j,0])/(2*XSpacing)
                strain_yy = (self.VPred[rgrs][i,j+1,1]-self.VPred[rgrs][i,j-1,1])/(2*YSpacing)
                self.ContMetric[rgrs].extend([-(strain_xx + strain_yy)])
            
            #Calculation of continuity metric (-S_xx-S_yy) for the SVM-Predicted field at the evaluation locations
#            strain_xx_SVR = (self.VPredSVR[i+1,j,0]-self.VPredSVR[i-1,j,0])/(2*XSpacing)
#            strain_yy_SVR = (self.VPredSVR[i,j+1,1]-self.VPredSVR[i,j-1,1])/(2*YSpacing)
#            self.ContMetricPredSVR.extend([-(strain_xx_SVR + strain_yy_SVR)])
#            
#            #Calculation of continuity metric (-S_xx-S_yy) for the MLP-Predicted field at the evaluation locations
#            strain_xx_MLP = (self.VPredMLP[i+1,j,0]-self.VPredMLP[i-1,j,0])/(2*XSpacing)
#            strain_yy_MLP = (self.VPredMLP[i,j+1,1]-self.VPredMLP[i,j-1,1])/(2*YSpacing)
#            self.ContMetricPredMLP.extend([-(strain_xx_MLP + strain_yy_MLP)])
#            
#            #Calculation of continuity metric (-S_xx-S_yy) for the GPOD-Predicted field at the evaluation locations
#            strain_xx_GPOD = (self.UPOD[i+1,j,0]-self.UPOD[i-1,j,0])/(2*XSpacing)
#            strain_yy_GPOD = (self.UPOD[i,j+1,1]-self.UPOD[i,j-1,1])/(2*YSpacing)
#            self.ContMetricPredGPOD.extend([-(strain_xx_GPOD + strain_yy_GPOD)])
        
        self.UMetric={}
        self.UMetric['Orig'] = []
        for rgrs in SelectedRegressors:
            self.UMetric[rgrs] = []
#        self.UMetricOrig = []
#        self.UMetricSVR = []
#        self.UMetricMLP = []
#        self.UMetricGPOD = []
        for sample in self.ContEvalsLoc:
            i = sample[0]
            j = sample[1]
            self.UMetric['Orig'].extend([self.VOrig[i,j,1]])
            for rgrs in SelectedRegressors:
                self.UMetric[rgrs].extend([self.VPred[rgrs][i,j,1]])
#            self.UMetricOrig.extend([self.VOrig[i,j,1]])
#            self.UMetricSVR.extend([self.VPredSVR[i,j,1]])
#            self.UMetricMLP.extend([self.VPredMLP[i,j,1]])
#            self.UMetricGPOD.extend([self.UPOD[i,j,1]])
        
        if SnapIndex in UID.SnapForPresentation:
            import matplotlib.pyplot as plt
            # Plotting the results:
            CBE_ScatterPlotsRegressors = ['SVR', 'MLP', 'MMI']
            Colors = ['y', 'g', 'c']
            plt.figure(figsize=(5,5))
            counter=0
            for PlotSelector in CBE_ScatterPlotsRegressors:
                plt.scatter(self.ContMetric['Orig'], self.ContMetric[PlotSelector], c=Colors[counter]
                        , alpha=0.75, label=PlotSelector+'-ORIG')
                counter+=1
    #        plt.scatter(self.ContMetric['Orig'], self.ContMetricPredSVR, c='y'
    #            , alpha=0.75, label='SVR-ORIG')
    #        plt.scatter(self.ContMetricOrig, self.ContMetricPredMLP, c='g'
    #            , alpha=0.6, label='MLP-ORIG')
    #        if UID.MLPlotStatus['GPOD'] == True:
    #            plt.scatter(self.ContMetricOrig, self.ContMetricPredGPOD, c='c'
    #                        , alpha=0.6, label='GPOD-ORIG')
            from sklearn.metrics import r2_score
            R2_SCORE = {}
            Title = ''
            for PlotSelector in CBE_ScatterPlotsRegressors:
                R2_SCORE[PlotSelector] = r2_score(self.ContMetric['Orig'], self.ContMetric[PlotSelector])
                Title += PlotSelector + ' R-Squared={}\n'.format(R2_SCORE[PlotSelector])
            plt.title(Title)
    #        SVR_r2_score = r2_score(self.ContMetricOrig, self.ContMetricPredSVR)
    #        MLP_r2_score = r2_score(self.ContMetricOrig, self.ContMetricPredMLP)
    #        GPOD_r2_score = r2_score(self.ContMetricOrig, self.ContMetricPredGPOD)
            
            
    #        if UID.MLPlotStatus['GPOD'] == True:
    #            plt.title('SVR R-Squared={}\nMLP R-Squared={}\nGPOD R-Squared={}'
    #                      .format(SVR_r2_score, MLP_r2_score, GPOD_r2_score))
    #        else:
    #            plt.title('SVR R-Squared={}\nMLP R-Squared={}'
    #                      .format(SVR_r2_score, MLP_r2_score))
            plt.xlabel('ORIG VEL')
            plt.ylabel('Predicted VEL')
            plt.legend()
            plt.plot([-4000,4000],[-4000,4000])
            
        
    
    def PredictionErrors(self, SelectedRegressors):
        import numpy as np
        self.ContMetricArray = {}
        self.ContAbsDeviation = {}
        self.ContMetricArray['Orig'] = np.array(self.ContMetric['Orig'])
        for rgrs in SelectedRegressors:
            self.ContMetricArray[rgrs] = np.array(self.ContMetric[rgrs])
            self.ContAbsDeviation[rgrs] = np.abs(self.ContMetricArray['Orig'] - self.ContMetricArray[rgrs])
#        self.ContMetricOrigArray = np.array(self.ContMetricOrig)
#        self.ContMetricSVRArray = np.array(self.ContMetricPredSVR)
#        self.ContMetricMLPArray = np.array(self.ContMetricPredMLP)
#        self.ContMetricGPODArray = np.array(self.ContMetricPredGPOD)
        
        
        
#        self.ContAbsDeviationSVR = np.abs(self.ContMetricOrigArray - self.ContMetricSVRArray)
#        self.ContAbsDeviationMLP = np.abs(self.ContMetricOrigArray - self.ContMetricMLPArray)
#        self.ContAbsDeviationGPOD = np.abs(self.ContMetricOrigArray - self.ContMetricGPODArray)
        
        #import matplotlib.pyplot as plt
        #plt.figure()
        