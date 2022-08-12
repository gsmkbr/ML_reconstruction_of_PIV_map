class VelReadingPreprocess:
    def __init__(self, VelMapProp):
        #self.FOV_No = VelMapProp.FOV_No
        self.NRow = VelMapProp.NRow
        self.NCol = VelMapProp.NCol
        #self.Snap_No = VelMapProp.InstSnap_No
        self.NoComponents = VelMapProp.NoComponents
        #self.StartSnapIndex = VelMapProp.StartSnapIndex
        #self.EndSnapIndex = VelMapProp.EndSnapIndex
        
        #NSanpshots = VelMapProp.EndSnapIndex - VelMapProp.StartSnapIndex + 1
        
            
    def FileNameGenerator(self, FOVIndex, SnapIndex):
        from math import log10, floor
        if floor(log10(SnapIndex+1)) == 0:
            self.FileName='..\\data\\FOV'+str(FOVIndex+1)+'_75_64pix_3D_000'+str(SnapIndex+1)+'.txt'
        elif floor(log10(SnapIndex+1)) == 1:
            self.FileName='..\\data\\FOV'+str(FOVIndex+1)+'_75_64pix_3D_00'+str(SnapIndex+1)+'.txt'
        elif floor(log10(SnapIndex+1)) == 2:
            self.FileName='..\\data\\FOV'+str(FOVIndex+1)+'_75_64pix_3D_0'+str(SnapIndex+1)+'.txt'
        elif floor(log10(SnapIndex+1)) == 3:
            self.FileName='..\\data\\FOV'+str(FOVIndex+1)+'_75_64pix_3D_'+str(SnapIndex+1)+'.txt'
        
    def ReadData(self, SnapIndex, VelMapProp):
        import numpy as np
        if SnapIndex == VelMapProp.StartSnapIndex-1:
            self.VOrig = np.zeros((VelMapProp.NoSnapshots, self.NRow[str(VelMapProp.FOVIndex)], self.NCol[str(VelMapProp.FOVIndex)], self.NoComponents))
            self.VOrigDuplicate = np.zeros((VelMapProp.NoSnapshots, self.NRow[str(VelMapProp.FOVIndex)], self.NCol[str(VelMapProp.FOVIndex)], self.NoComponents))
        VMat = np.loadtxt(self.FileName)   
        self.VOrig[SnapIndex,:,:,:] = VMat.reshape(1, self.NRow[str(VelMapProp.FOVIndex)], self.NCol[str(VelMapProp.FOVIndex)], self.NoComponents)
        self.VOrigDuplicate[SnapIndex,:,:,:] = self.VOrig[SnapIndex,:,:,:]
    
#    def VelGradinets(self, VelMapProp, SnapIndex, nGradFeat_x, nGradFeat_y):
    
    def VelocityMask(self, SnapIndex, VelMapProp, UID):
        from math import sqrt
        NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        for i in range(0,NRow):
            for j in range(0,NCol):
                if UID.FOVIndex==1:
                    y_rotor = -143+sqrt(169**2-j**2)
                    if i<y_rotor:
                        self.VOrig[SnapIndex,i,j,:]=0
                        self.VOrigDuplicate[SnapIndex,i,j,:]=0
                elif UID.FOVIndex==2:
                    y_rotor = -143+sqrt(169**2-(j-(NCol-1))**2)
                    if i<y_rotor:
                        self.VOrig[SnapIndex,i,j,:]=0
                        self.VOrigDuplicate[SnapIndex,i,j,:]=0
        
    def OrigGappiness(self, SnapIndex, VelMapProp, UID):
        import numpy as np
        if SnapIndex==VelMapProp.StartSnapIndex-1:
            self.OutRotorNullPercentage = []
        Matrix = self.VOrigDuplicate[SnapIndex,:,:,0].copy()
        NumElements = Matrix.size
        Matrix[np.where(Matrix!=0)]=1
        TotalNullPercentage = 100 - np.sum(Matrix)/NumElements*100
        OutRotorNullPercentage = TotalNullPercentage - UID.MaskPercent
        self.OutRotorNullPercentage.append((SnapIndex, OutRotorNullPercentage))
    
    def VelFiltered(self, VelMapProp, SnapIndex, FiltLenX, FiltLenY, FiltType):
        import numpy as np
        self.nFiltSizes = np.array(FiltLenX).size
        NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        if SnapIndex == VelMapProp.StartSnapIndex-1:
            self.VGappyFiltered = np.zeros((self.nFiltSizes, VelMapProp.NoSnapshots, NRow, NCol, self.NoComponents))
        for iteration in range (0, self.nFiltSizes):
            X_margin = int((FiltLenX[iteration]-1)/2)
            LeftBound = X_margin
            RightBound = NCol - X_margin
            Y_margin = int((FiltLenY[iteration]-1)/2)
            BottomBound = Y_margin
            TopBound = NRow - Y_margin
            
            FiltBoxSize = (FiltLenX[iteration], FiltLenY[iteration])
            if FiltType == 'BoxUniform':
                FiltCoeffs = np.ones(FiltBoxSize)
            elif FiltType == 'Gaussian':
                sigma = 3
                xRange = np.arange(-X_margin,X_margin+1)
                xRange = np.tile(xRange,(FiltLenY[iteration],1))
                yRange = np.arange(-Y_margin,Y_margin+1)
                yRange = np.tile(yRange,(FiltLenX[iteration],1)).T
                FiltCoeffs = 1/(2*np.pi*sigma**2)*np.exp(-(np.power(xRange,2)+np.power(yRange,2))/(2*sigma**2))
            for Comp in range (0, self.NoComponents):
                for i in range (BottomBound, TopBound):
                    for j in range (LeftBound, RightBound):
                        TempArray = self.VOrig[SnapIndex, i-X_margin:i+X_margin+1, j-Y_margin:j+Y_margin+1, Comp]
                        ZeroIndex = np.where(TempArray==0)
                        FiltCoeffsGappy = np.copy(FiltCoeffs)
                        FiltCoeffsGappy[ZeroIndex] = 0
                        ProductMat = np.multiply(TempArray, FiltCoeffsGappy)
                        AveInFiltBox = ProductMat.sum()/FiltCoeffsGappy.sum()
                        self.VGappyFiltered[iteration, SnapIndex, i, j, Comp] = AveInFiltBox
            
        #substituting NaN with Zero
        self.VGappyFiltered[np.where(np.isnan(self.VGappyFiltered))]=0
    
    def FiltVorticity(self, VelMapProp, SnapIndex, XSpacing, YSpacing):
        import numpy as np
        NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        
        if SnapIndex == VelMapProp.StartSnapIndex-1:
            self.FilteredVorticity = np.zeros((self.nFiltSizes, VelMapProp.NoSnapshots, NRow, NCol))
        for iteration in range (0, self.nFiltSizes):
            for Comp in range (0, self.NoComponents):
                V = np.copy(self.VGappyFiltered[iteration, SnapIndex, : , : , 0:2])
                for i in range (1, NRow-1):
                    for j in range (1, NCol-1):
                        if V[i, j+1, 1]!=0 and V[i, j-1, 1]!=0 and V[i+1,j,0]!=0 and V[i-1,j,0]!=0:
                            Term1 = (V[i,j+1,1]-V[i,j-1,1])/(2*XSpacing)
                            Term2 = (V[i+1,j,0]-V[i-1,j,0])/(2*YSpacing)
                            self.FilteredVorticity[iteration, SnapIndex, i, j] = Term1 - Term2
    
    def FiltStrain(self, VelMapProp, SnapIndex, XSpacing, YSpacing):
        import numpy as np
        NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        
        if SnapIndex == VelMapProp.StartSnapIndex-1:
            self.FilteredStrain = np.zeros((self.nFiltSizes, VelMapProp.NoSnapshots, NRow, NCol, 3))
        for iteration in range (0, self.nFiltSizes):
            V = np.copy(self.VGappyFiltered[iteration, SnapIndex, : , : , 0:2])
            for i in range (1, NRow-1):
                for j in range (1, NCol-1):
                    if V[i, j+1, 1]!=0 and V[i, j-1, 1]!=0:
                        self.FilteredStrain[iteration, SnapIndex, i, j, 0] = (V[i,j+1,1]-V[i,j-1,1])/(2*XSpacing)
                    if V[i+1,j,0]!=0 and V[i-1,j,0]!=0:
                        self.FilteredStrain[iteration, SnapIndex, i, j, 1] = (V[i+1,j,0]-V[i-1,j,0])/(2*YSpacing)
                    if V[i, j+1, 1]!=0 and V[i, j-1, 1]!=0 and V[i+1,j,0]!=0 and V[i-1,j,0]!=0:
                        Term1 = (V[i+1,j,0]-V[i-1,j,0])/(2*YSpacing)
                        Term2 = (V[i,j+1,1]-V[i,j-1,1])/(2*XSpacing)
                        self.FilteredStrain[iteration, SnapIndex, i, j, 2] = 0.5*(Term1 + Term2)
                        
    def ConvertToDataFrame(self, ActiveFeat, VelMapProp, SnapIndex, TargetComponent=0, SCLMethod = 'StandardScaler'):
        import pandas as pd
        count=0
        w=[]
        #step is a parameter for sparse sampling from FOVs
        step = 1
        for i in range (0,self.NRow[str(VelMapProp.FOVIndex)],step):
            for j in range (0,self.NCol[str(VelMapProp.FOVIndex)],step):
                if self.VOrig[SnapIndex,i,j,0]!=0:
                    count+=1
                    
                    Position = [i+1,j+1]
                    VGappyFiltered = []
                    VorticityFiltered = []
                    StrainFiltered = []
                    for FiltSizeIndex in range (0,self.nFiltSizes):
                        VGappyFiltered.extend([self.VGappyFiltered[FiltSizeIndex,SnapIndex,i,j,TargetComponent]])
                    for FiltSizeIndex in range (0,self.nFiltSizes):
                        VorticityFiltered.extend([self.FilteredVorticity[FiltSizeIndex,SnapIndex,i,j]])
                    for FiltSizeIndex in range (0,self.nFiltSizes):
                        for comp in range(0,3):
                            StrainFiltered.extend([self.FilteredStrain[FiltSizeIndex,SnapIndex,i,j,comp]])
                    TargetQuantity = [self.VOrig[SnapIndex,i,j,TargetComponent]]
                    ActiveFeat.SelectedFeatures(Position=Position, VGappyFiltered=VGappyFiltered, VorticityFiltered=VorticityFiltered, StrainFiltered=StrainFiltered, TargetQuantity=TargetQuantity)
                    
                    ##TempList = []
                    ##TempList.extend([i+1,j+1,self.VGappyFiltered[0,SnapIndex,i,j,TargetComponent],self.VGappyFiltered[1,SnapIndex,i,j,TargetComponent]])
                    #for Comp in range (0, VelMapProp.NoComponents):
                    #    TempList.extend(self.VGappyFiltered[:,SnapIndex,i,j,Comp])
#                    if VelMapProp.NoComponents == 3:
#                        TempList.extend([self.VOrig[SnapIndex,i,j,0],self.VOrig[SnapIndex,i,j,1],self.VOrig[SnapIndex,i,j,2]])
#                    else:
#                        TempList.extend([self.VOrig[SnapIndex,i,j,0],self.VOrig[SnapIndex,i,j,1]])
                    ##TempList.extend([self.VOrig[SnapIndex,i,j,TargetComponent]])
                    ##w.append(TempList)
                    w.append(ActiveFeat.SelFeatures)
        
        # Scaling the input data
        SelectedDF=pd.DataFrame(w)
        InputDF = SelectedDF.drop(SelectedDF.columns[-1], axis=1, inplace=False)
        TargetDF = SelectedDF[SelectedDF.columns[-1]]
        if SCLMethod == 'None':
            InputDFScaled = InputDF
        else:
            if SCLMethod == 'StandardScaler':
                from sklearn.preprocessing import StandardScaler
                self.Scaler = StandardScaler().fit(InputDF)
            elif SCLMethod == 'MinMaxScaler':
                from sklearn.preprocessing import MinMaxScaler
                self.Scaler = MinMaxScaler(feature_range=(0,1)).fit(InputDF)
            elif SCLMethod == 'RobustScaler':
                from sklearn.preprocessing import RobustScaler
                self.Scaler = RobustScaler(quantile_range=(25.0,75.0)).fit(InputDF)
            elif SCLMethod == 'Normalizer':
                from sklearn.preprocessing import Normalizer
                self.Scaler = Normalizer().fit(InputDF)
            InputDFScaled = self.Scaler.transform(InputDF)
            InputDFScaled = pd.DataFrame(InputDFScaled)
        
        self.VOrigDF=InputDFScaled.copy()
        self.VOrigDF[SelectedDF.columns[-1]]=TargetDF
        
        self.NOutputFeatures = 1
        self.NInputFeatures = self.VOrigDF.shape[1]-self.NOutputFeatures
        
        
    def InputOutputSplit(self, TargetComponent=0, TestSize=0.0):
        # TargetComponent=0, 1 and 2 for the x, y and z
        # velocity components, respectively
        inputs=self.VOrigDF.loc[:,0:self.NInputFeatures-1]
        output=self.VOrigDF.loc[:,self.NInputFeatures]
        # SPLITING DATA INTO TRAIN AND TEST PORTIONS
        from sklearn.model_selection import train_test_split
        RandState = 55
        self.InputTrain,self.InputTest,self.TargetTrain,self.TargetTest=train_test_split(inputs,output,test_size=TestSize,random_state=RandState)
        
    