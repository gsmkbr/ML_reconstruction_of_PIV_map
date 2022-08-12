class MakeGappy:
    def __init__(self, VelMapProp, GapSize = [2,4], GappinessPercent = 25):
        self.NRow = VelMapProp.NRow
        self.NCol = VelMapProp.NCol
        self.GapSize = GapSize
        self.GappinessPercent = GappinessPercent
        
    def GapGenerator(self, V_Orig, VelMapProp, SnapIndex, UID):
        import numpy as np
        
        # A function to calculate the gappiness percent of data for 
        # a matrix
        def GappinessPercentCalculator(Matrix):
            ZeroMatrix = np.where(Matrix == 0)
            NonZeroMatrix = np.where(Matrix != 0)
            Mat_temp = np.copy(Matrix)
            Mat_temp[ZeroMatrix] = 1
            Mat_temp[NonZeroMatrix] = 0
            GapPercent = np.sum(Mat_temp)/np.size(Mat_temp)
            return GapPercent
        
        # Detecting the location of non-zero velocities in the original
        # velocity map
        NonZeroVOrig = V_Orig.VOrig[SnapIndex,:,:,:] != 0
        
        # Calculation of gappiness percent of the original velocity
        # field and random increase of gappy spots, until reaching 
        # a specified level of gappiness
        GapPercent = GappinessPercentCalculator(V_Orig.VOrig[SnapIndex,:,:,0])
        from random import randint
        while (GapPercent < self.GappinessPercent/100):
            LocalGapSize = randint(self.GapSize[0],self.GapSize[1])
            GapXLocation = randint(0,self.NCol[str(VelMapProp.FOVIndex)]-LocalGapSize)
            GapYLocation = randint(0,self.NRow[str(VelMapProp.FOVIndex)]-LocalGapSize)
            V_Orig.VOrig[SnapIndex,GapYLocation:GapYLocation+LocalGapSize,GapXLocation:GapXLocation+LocalGapSize,:]=0
            GapPercent = GappinessPercentCalculator(V_Orig.VOrig[SnapIndex,:,:,0])
        
        # Detecting the location of zero velocities in the artificially-gappy
        # velocity map
        ZeroVGappy = V_Orig.VOrig[SnapIndex,:,:,:] == 0
        # Detecting locations where the original velocity
        # is non-zero but the artificial-gappy field is zero (in order
        # to evaluate the accuracy of model)
        if SnapIndex == VelMapProp.StartSnapIndex-1:
            self.LocationsForEvaluation = np.zeros((VelMapProp.EndSnapIndex, self.NRow[str(VelMapProp.FOVIndex)], self.NCol[str(VelMapProp.FOVIndex)], V_Orig.NoComponents), dtype = bool)
        self.LocationsForEvaluation[SnapIndex,:,:,:] = np.logical_and(NonZeroVOrig, ZeroVGappy)
        
        #masking the locations for evaluation indlucing:
        # the Boundary edge layers
        # the rotor region (in FOVs 1 and 2) which is previously filtered
        from math import floor
        NullEdgeWidthX = floor(UID.FiltLenX[-1]/2)
        NullEdgeWidthY = floor(UID.FiltLenY[-1]/2)
        NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        # filtering left and right vertical filtered boundary
        self.LocationsForEvaluation[SnapIndex,:,0:NullEdgeWidthX+1,:]=False
        self.LocationsForEvaluation[SnapIndex,:,NCol-NullEdgeWidthX:NCol,:]=False
        # filtering bottom and top horizontal filtered boundary
        self.LocationsForEvaluation[SnapIndex,0:NullEdgeWidthY+1,:,:]=False
        self.LocationsForEvaluation[SnapIndex,NRow-NullEdgeWidthY:NRow,:,:]=False
        