class reconsByInterpolation:
    def __init__(self, FOVIndex, V_Orig, VelMapProp):
        self.NRow = V_Orig.NRow[str(FOVIndex-1)]
        self.NCol = V_Orig.NCol[str(FOVIndex-1)]
        import numpy as np
        self.VNearestND = np.zeros((VelMapProp.NoSnapshots, self.NRow, self.NCol, VelMapProp.NoComponents))
        self.VLinearND = np.zeros((VelMapProp.NoSnapshots, self.NRow, self.NCol, VelMapProp.NoComponents))
        self.VGridLinear = np.zeros((VelMapProp.NoSnapshots, self.NRow, self.NCol, VelMapProp.NoComponents))
        self.VGridCubic = np.zeros((VelMapProp.NoSnapshots, self.NRow, self.NCol, VelMapProp.NoComponents))
        self.VMovingMedian = np.zeros((VelMapProp.NoSnapshots, self.NRow, self.NCol, VelMapProp.NoComponents))
        
    def GeneralParameters(self, V_Orig, SnapIndex, PredictedComponent):
        import numpy as np
        self.ColIndex = np.arange(0,self.NCol)
        self.RowIndex = np.arange(0,self.NRow)
        VOrig2D = V_Orig.VOrig[SnapIndex,:,:,PredictedComponent]
#        GridColIndices = np.linspace(min(ColIndex), max(ColIndex))
#        GridRowIndices = np.linspace(min(RowIndex), max(RowIndex))
#        GridColIndices, GridRowIndices = np.meshgrid(GridColIndices, GridRowIndices)
        self.GridColIndices, self.GridRowIndices = np.meshgrid(self.ColIndex, self.RowIndex)
        self.InputPoints = []
        self.TargetVelocity = []
        from math import nan
        for i in self.RowIndex:
            for j in self.ColIndex:
                if VOrig2D[i,j]!=0 and VOrig2D[i,j]!=nan:
                    self.InputPoints.extend([(j,i)])
                    self.TargetVelocity.extend([VOrig2D[i,j]])
        
        
    def NearestNDInterpolator(self, SnapIndex, PredictedComponent):
        from scipy.interpolate import NearestNDInterpolator
        import numpy as np
        self.interpolator = NearestNDInterpolator(self.InputPoints, self.TargetVelocity)
        InputPointsForInterpolation = []
        for i in self.RowIndex:
            for j in self.ColIndex:
                InputPointsForInterpolation.extend([[j,i]])
                self.VNearestND[SnapIndex,i,j,PredictedComponent] = self.interpolator((i,j))
        #self.VNearestND[SnapIndex,:,:,PredictedComponent] = interpolator(self.GridColIndices,self.GridRowIndices)
        #self.VNearestND[SnapIndex,:,:,PredictedComponent] = interpolator(InputPointsForInterpolation)
        
    def GridInterpolation(self, SnapIndex, PredictedComponent):
        from scipy.interpolate import griddata
        import numpy as np
        self.VGridLinear[SnapIndex,:,:,PredictedComponent] = griddata(self.InputPoints, self.TargetVelocity, (self.GridColIndices, self.GridRowIndices), method='linear')
        self.VGridLinear[np.where(np.isnan(self.VGridLinear))]=0
        self.VGridCubic[SnapIndex,:,:,PredictedComponent] = griddata(self.InputPoints, self.TargetVelocity, (self.GridColIndices, self.GridRowIndices), method='cubic')
        self.VGridCubic[np.where(np.isnan(self.VGridCubic))]=0
        #self.VGridNearestNeigh[SnapIndex,:,:,PredictedComponent] = griddata(self.InputPoints, self.TargetVelocity, (self.GridColIndices, self.GridRowIndices), method='nearest')
        #self.VGridNearestNeigh[np.where(np.isnan(self.VGridNearestNeigh))]=0
        
    def MovingMedianInterpolation(self, VelMapProp, V_Orig, SnapIndex, BoxLengthX, BoxLengthY):
        import numpy as np
        if SnapIndex == VelMapProp.StartSnapIndex-1:
            self.VGappyFiltered = np.zeros((VelMapProp.NoSnapshots, self.NRow, self.NCol, VelMapProp.NoComponents))
        X_margin = int((BoxLengthX-1)/2)
        LeftBound = X_margin
        RightBound = self.NCol - X_margin
        Y_margin = int((BoxLengthY-1)/2)
        BottomBound = Y_margin
        TopBound = self.NRow - Y_margin
            
#        FiltBoxSize = (FiltLenX[iteration], FiltLenY[iteration])
#        if FiltType == 'BoxUniform':
#            FiltCoeffs = np.ones(FiltBoxSize)
#        elif FiltType == 'Gaussian':
#            sigma = 3
#            xRange = np.arange(-X_margin,X_margin+1)
#            xRange = np.tile(xRange,(FiltLenY[iteration],1))
#            yRange = np.arange(-Y_margin,Y_margin+1)
#            yRange = np.tile(yRange,(FiltLenX[iteration],1)).T
#            FiltCoeffs = 1/(2*np.pi*sigma**2)*np.exp(-(np.power(xRange,2)+np.power(yRange,2))/(2*sigma**2))
        from math import isnan
        #initialization of the field
        self.VMovingMedian[SnapIndex,:,:,:] = V_Orig.VOrig[SnapIndex,:,:,:]
        for Comp in range (0, VelMapProp.NoComponents):
            for i in range (BottomBound, TopBound):
                for j in range (LeftBound, RightBound):
                    TempArray = V_Orig.VOrig[SnapIndex, i-X_margin:i+X_margin+1, j-Y_margin:j+Y_margin+1, Comp]
                    if isnan(np.median(TempArray[np.where(TempArray!=0)])):
                        self.VMovingMedian[SnapIndex, i, j, Comp] = 0
                    else:
                        self.VMovingMedian[SnapIndex, i, j, Comp] = np.median(TempArray[np.where(TempArray!=0)])