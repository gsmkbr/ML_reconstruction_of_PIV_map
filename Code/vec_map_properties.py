class VMapProp:
    def __init__(self, FOVIndex, No_of_FOVs, InstSnapIndex, NoComponents, StartSnapIndex, EndSnapIndex):
        #self.FOV_No = FOV_No
        #self.NRow = NRow
        #self.NCol = NCol
        self.InstSnapIndex = InstSnapIndex-1
        self.NoComponents = NoComponents
        self.StartSnapIndex = StartSnapIndex
        self.EndSnapIndex = EndSnapIndex
        self.FOVIndex = FOVIndex-1
        self.No_of_FOVs = No_of_FOVs
        self.NoSnapshots = self.EndSnapIndex - self.StartSnapIndex + 1
        self.NRow = {}
        self.NCol = {}
        for FOV_Index in range (0, No_of_FOVs):
            if FOV_Index==0:
                self.NRow[str(FOV_Index)] = 72
                self.NCol[str(FOV_Index)] = 97
            elif FOV_Index==1:
                self.NRow[str(FOV_Index)] = 73
                self.NCol[str(FOV_Index)] = 98
            elif FOV_Index==2:
                self.NRow[str(FOV_Index)] = 71
                self.NCol[str(FOV_Index)] = 97
            elif FOV_Index==3:
                self.NRow[str(FOV_Index)] = 71
                self.NCol[str(FOV_Index)] = 97