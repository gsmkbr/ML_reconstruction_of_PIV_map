def SaveData(Data, FileName):
    import pickle
    file = open(FileName, "wb")
    pickle.dump(Data, file)
    file.close()

def LoadData(FileName):
    import pickle
    file = open(FileName, "rb")
    output = pickle.load(file)
    return output

SavingStatus = False
#DataToSave, FileName = ContMetricSVR2.copy(), 'ContMetricSVR.pkl'
LoadingStatus = True

if SavingStatus:
    SaveData(ContAbsDeviationSVR2.copy(), 'ContAbsDeviationSVR.pkl')
    SaveData(ContAbsDeviationMLP2.copy(), 'ContAbsDeviationMLP.pkl')
    SaveData(ContMetricOrig2.copy(), 'ContMetricOrig.pkl')
    SaveData(ContMetricMLP2.copy(), 'ContMetricMLP.pkl')
    SaveData(ContMetricSVR2.copy(), 'ContMetricSVR.pkl')
    
if LoadingStatus:
    ContAbsDeviationSVR3 = LoadData('ContAbsDeviationSVR.pkl')
    ContAbsDeviationMLP3 = LoadData('ContAbsDeviationMLP.pkl')
    ContMetricOrig3 = LoadData('ContMetricOrig.pkl')
    ContMetricMLP3 = LoadData('ContMetricMLP.pkl')
    ContMetricSVR3 = LoadData('ContMetricSVR.pkl')