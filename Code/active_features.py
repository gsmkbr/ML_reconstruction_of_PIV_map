class ActiveFeatures:
    def __init__(self, Position=True, FilteredVel=True, FilteredVort=True, FilteredStrain = True):
        self.ActiveKwrds = {'Position':Position, 'FilteredVel':FilteredVel, 'FilteredVort':FilteredVort, 'FilteredStrain':FilteredStrain}
    
    def SelectedFeatures(self, Position, VGappyFiltered, VorticityFiltered, StrainFiltered, TargetQuantity):
        self.SelFeatures = []
        if self.ActiveKwrds['Position'] == True:
            self.SelFeatures.extend(Position)
        if self.ActiveKwrds['FilteredVel'] == True:
            self.SelFeatures.extend(VGappyFiltered)
        if self.ActiveKwrds['FilteredVort'] == True:
            self.SelFeatures.extend(VorticityFiltered)
        if self.ActiveKwrds['FilteredStrain'] == True:
            self.SelFeatures.extend(StrainFiltered)
        if TargetQuantity != False:
            self.SelFeatures.extend(TargetQuantity)