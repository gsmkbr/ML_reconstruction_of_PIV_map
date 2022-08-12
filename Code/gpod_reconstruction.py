class GappyPOD:
    def __init__(self, UForPODOrig, VelMapProp, nMode, MaxIter):
        self.StartSnapshotIndex = VelMapProp.StartSnapIndex
        self.EndSnapshotIndex = VelMapProp.EndSnapIndex
        self.nMode = nMode
        self.MaxIter = MaxIter
        self.NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        self.NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        self.NoSnapshots = VelMapProp.NoSnapshots
        self.NoComponents = VelMapProp.NoComponents
        self.UForPODOrig = UForPODOrig
#        from vec_map_properties import VMapProp
#        self.UForPODOrig=[]
#        for i in range(StartSnapshotIndex, EndSnapshotIndex+1):
#            VelMapProp = VMapProp(FOV_No=V_Reference.FOV_No, NRow=V_Reference.NRow, NCol=V_Reference.NCol, Snap_No=i, NoComponents=V_Reference.NoComponents)
#            from velocity_reading_and_preprocessing import VelReadingPreprocess
#            V_Orig = VelReadingPreprocess(VelMapProp)
#            V_Orig.FileNameGenerator()
#            V_Orig.ReadData()
#            self.UForPODOrig.append(V_Orig.VOrig)
#        import numpy as np
#        self.UForPODOrig = np.array(self.UForPODOrig)
        
    def DefiningMask(self):
        import numpy as np
        self.Mu = np.copy(self.UForPODOrig[:,:,:,0])
        NonZeroIndex = np.where(self.Mu != 0)
        self.Mu[NonZeroIndex]=1
        
    
    def VelAverage(self):
        import numpy as np
        #Calculating the ensemble average of velocity vector maps
        NonZeroIndex = np.where(self.UForPODOrig!=0)
        NonZeroBool = np.copy(self.UForPODOrig)
        NonZeroBool[NonZeroIndex]=1
        self.UAve = np.zeros((self.NRow, self.NCol, self.NoComponents))
        for component in range(0,self.NoComponents):
            self.UAve[:,:,component] = np.divide(self.UForPODOrig.sum(0)[:,:,component], NonZeroBool.sum(0)[:,:,component])
        self.UAve[np.where(np.isnan(self.UAve))]=0
        
        #Substituting non-existing velocity data with that of averaged velocity
        self.UPOD = np.zeros(self.UForPODOrig.shape)
        for SnapIndex in range(self.StartSnapshotIndex-1, self.EndSnapshotIndex):
            TempVecMap = np.copy(self.UForPODOrig[SnapIndex,:,:,:])
            #NaNIndex = np.where(np.isnan(self.UForPODOrig[SnapIndex,:,:,:]))
            ZeroIndex = np.where(self.UForPODOrig[SnapIndex,:,:,:]==0)
            #TempVecMap[NaNIndex] = self.UAve[NaNIndex]
            TempVecMap[ZeroIndex] = self.UAve[ZeroIndex]
            self.UPOD[SnapIndex,:,:,:] = TempVecMap
            
    def CorrelationMatrix(self):
        import numpy as np
        self.CorrMat = np.zeros((self.NoSnapshots, self.NoSnapshots))
        for k in range(0, self.NoSnapshots):
            for l in range(0, self.NoSnapshots):
                USquared=np.multiply(self.UPOD[k,:,:,:],self.UPOD[l,:,:,:])
                self.CorrMat[k,l]=np.sum(USquared)/self.NoSnapshots
        
    def SolEigVal(self):
        import numpy as np
#        self.EigVal, self.EigVec = np.linalg.eig(self.CorrMat)
#        self.EigVal = np.diag(self.EigVal)
#        #sorting eigenvalues and the corresponding eigenvectors ascending
#        self.EigVec = np.fliplr(self.EigVec)
#        self.EigVal = np.flipud(np.fliplr(self.EigVal))
        
        self.EigVec, self.EigVal, TempMat = np.linalg.svd(self.CorrMat)
        
    def BasisFunc(self):
        import numpy as np
        self.phi = np.zeros((self.nMode, self.NRow, self.NCol, self.NoComponents))
        for sIndx in range (0, self.nMode):
            for k in range (0, self.NoSnapshots):
                self.phi[sIndx,:,:,:] = self.phi[sIndx,:,:,:] + self.EigVec[k,sIndx]*self.UPOD[k,:,:,:]
        
    def CalcB(self):
        import numpy as np
        # M.b = V  (Equation 3.24 of Rahmani thesis, page 33)
        self.b = np.zeros((self.NoSnapshots,self.nMode))
        M = np.zeros((self.nMode,self.nMode))
        V = np.zeros((self.nMode,1))
        
        for k in range (0, self.NoSnapshots):
            for sModePrime in range (0, self.nMode):
                TempMat=np.multiply(
                        (np.multiply(self.UPOD[k,:,:,0], self.phi[sModePrime,:,:,0])
                       + np.multiply(self.UPOD[k,:,:,1], self.phi[sModePrime,:,:,1])
                       + np.multiply(self.UPOD[k,:,:,2], self.phi[sModePrime,:,:,2]))
                        , self.Mu[k,:,:])
                V[sModePrime,0] = np.sum(TempMat)
                for sMode in range (0, self.nMode):
                    TempMat=np.multiply(
                            (np.multiply(self.phi[sMode,:,:,0], self.phi[sModePrime,:,:,0])
                           + np.multiply(self.phi[sMode,:,:,1], self.phi[sModePrime,:,:,1])
                           + np.multiply(self.phi[sMode,:,:,2], self.phi[sModePrime,:,:,2]))
                            , self.Mu[k,:,:])
                    M[sModePrime,sMode] = np.sum(TempMat)
            
            SolutionForB = np.linalg.solve(M, V)
            self.b[k,:] = SolutionForB.reshape(1, self.nMode)
            
    def VelocityUpdate(self):
        import numpy as np
        # Calculation of updated velocity field based on nMode modes: UPOD
        self.UPOD_EntireDomain=np.zeros((self.NoSnapshots, self.NRow, self.NCol, self.NoComponents))
        for k in range (0, self.NoSnapshots):
            for sIndx in range (0, self.nMode):
                self.UPOD_EntireDomain[k,:,:,:]=self.UPOD_EntireDomain[k,:,:,:]+self.b[k,sIndx]*self.phi[sIndx,:,:,:]
        
        # substituting updated velocity just in the gappy locations
        ZeroIndex = np.where(self.Mu == 0)
        for component in range (0, self.NoComponents):
            TempVecMap1 = np.copy(self.UPOD[:,:,:,component])
            TempVecMap2 = self.UPOD_EntireDomain[:,:,:,component]
            TempVecMap1[ZeroIndex] = TempVecMap2[ZeroIndex]
            self.UPOD[:,:,:,component] = TempVecMap1
            #self.UPOD[:,:,:,component] = UPOD_EntireDomain[:,:,:,component]
 
    def ErrorEvaluation(self, V_Orig, GappyProp, VelMapProp):
        import numpy as np
        self.PODError = 0
        for SnapIndex in range(VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
            for PredictedComponent in range (0,VelMapProp.NoComponents):
                VTemp1 = V_Orig.VOrig[SnapIndex,:,:,PredictedComponent]
                Target = VTemp1[np.array(GappyProp.LocationsForEvaluation[SnapIndex,:,:,PredictedComponent])]
                VTemp2 = self.UPOD[SnapIndex,:,:,PredictedComponent]
                #VTemp2 = self.UPOD_EntireDomain[SnapIndex,:,:,PredictedComponent]
                Prediction = VTemp2[np.array(GappyProp.LocationsForEvaluation[SnapIndex,:,:,PredictedComponent])]
                self.PODError += np.sum(np.sum(np.abs(np.subtract(Prediction,Target))))
                