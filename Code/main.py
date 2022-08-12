from time import time
StartTime = time()
from vec_map_properties import VMapProp
from velocity_reading_and_preprocessing import VelReadingPreprocess
from ml_models import DefiningMLModels
from reconst_by_interpolation import reconsByInterpolation
import numpy as np
from user_input_data import UserInputData
from continuity_based_evals import ContinuityBasedEvaluation
import matplotlib.pyplot as plt
UID = UserInputData()
PredDevForBoxPlot_C1 = []
PredDevForBoxPlot_C2 = []
PredDevForBoxPlot_C3 = []

ContAbsDeviation = {}
ContMetric = {}
# FOVs for the analysis:
FOVS = [2]
for FOVIndex in FOVS:
    # Setting the vector map properties by VMapProp Class
    UID.VelocityInputData(FOVIndex)
    # Generating an Excel file for the results
    import xlsxwriter
    XLSXResults = xlsxwriter.Workbook("Results_FOV "+str(UID.FOVIndex)+".xlsx")
    
    VelMapProp = VMapProp(FOVIndex=UID.FOVIndex, No_of_FOVs=UID.N_FOVs, InstSnapIndex=UID.InstSnapIndex, NoComponents=UID.NoComponents, StartSnapIndex=UID.StartSnapIndex, EndSnapIndex=UID.EndSnapIndex)
    #V_Orig is an object for the corresponding data of
    #the original velcoity vector map
    V_Orig = VelReadingPreprocess(VelMapProp)
    #for FOVIndex in range (0, VelMapProp.No_of_FOVs):
    
    from artificial_gappiness import MakeGappy
    UID.GappyProperties(VelMapProp)
    GappyProp = MakeGappy(VelMapProp, GapSize = UID.GapSize, GappinessPercent = UID.GappinessPercent)
    
    for SnapIndex in range (VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
        # Generating the file name to import into V_Orig.FileName
        V_Orig.FileNameGenerator(VelMapProp.FOVIndex, SnapIndex)
        # Reading of velocity data into the array V_Orig.VOrig
        V_Orig.ReadData(SnapIndex, VelMapProp)
        # Masking the rotor region
        UID.VelMaskProperties()
        UID.VelocityOperationsProp()
        if UID.VelMaskStatus==True and (UID.FOVIndex==1 or UID.FOVIndex==2):
            V_Orig.VelocityMask(SnapIndex, VelMapProp, UID)
        V_Orig.OrigGappiness(SnapIndex, VelMapProp, UID)
        GappyProp.GapGenerator(V_Orig, VelMapProp, SnapIndex, UID)
        V_Orig.VelFiltered(VelMapProp, SnapIndex, FiltLenX = UID.FiltLenX, FiltLenY = UID.FiltLenY, FiltType = UID.FiltType)
        V_Orig.FiltVorticity(VelMapProp, SnapIndex, XSpacing = UID.PIV_XSpacing, YSpacing = UID.PIV_YSpacing)
        V_Orig.FiltStrain(VelMapProp, SnapIndex, XSpacing = UID.PIV_XSpacing, YSpacing = UID.PIV_YSpacing)
        
    from gpod_reconstruction import GappyPOD
    UID.GappyPODProperties()
    V_POD = GappyPOD(V_Orig.VOrig, VelMapProp, nMode = UID.nMode, MaxIter = UID.MaxIter)
    V_POD.DefiningMask()
    V_POD.VelAverage()
    
    iteration = 0
    while iteration < V_POD.MaxIter:
        iteration +=1
        # calculation of correlation matrix C
        V_POD.CorrelationMatrix()
        # Solving the eigen value problem
        V_POD.SolEigVal()
        # Calculation of basis functions for different modes: Phi
        V_POD.BasisFunc()
        # calculation of coefficients b by solving system of equations
        V_POD.CalcB()
        # Calculation of updated velocity field based on nMode modes: UPOD
        V_POD.VelocityUpdate()
        
        V_POD.ErrorEvaluation(V_Orig, GappyProp, VelMapProp)
        print('POD Process: Iteration: {}  Total Error: {}'.format(iteration, V_POD.PODError))
    
    #Reconstruction of velocity field by various statistical interpolation methods
    for SnapIndex in range(VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
        if SnapIndex in UID.SnapshotsForNonGPODRegressors:
            for PredictedComponent in range (0,VelMapProp.NoComponents):
                if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
                    V_Interpol = reconsByInterpolation(FOVIndex, V_Orig, VelMapProp)
                V_Interpol.GeneralParameters(V_Orig, SnapIndex, PredictedComponent)
                V_Interpol.NearestNDInterpolator(SnapIndex, PredictedComponent)
                V_Interpol.GridInterpolation(SnapIndex, PredictedComponent)
            V_Orig.VelFiltered(VelMapProp, SnapIndex, FiltLenX = [7], FiltLenY = [7], FiltType = 'Gaussian')
            V_Interpol.VMovingAveGaussian = V_Orig.VGappyFiltered[0,:,:,:,:]
            V_Orig.VelFiltered(VelMapProp, SnapIndex, FiltLenX = [7], FiltLenY = [7], FiltType = 'BoxUniform')
            V_Interpol.VMovingAveUniform = V_Orig.VGappyFiltered[0,:,:,:,:]
            V_Interpol.MovingMedianInterpolation(VelMapProp, V_Orig, SnapIndex, BoxLengthX = 7, BoxLengthY = 7)
    #V_POD.UPOD = V_Interpol.VMovingMedian.copy()
    
    
    from active_features import ActiveFeatures
    #Specifying the active features in learning and prediction
    UID.ActiveFeaturesConfig()
    ActiveFeat = ActiveFeatures(Position=UID.PositionStatus, FilteredVel=UID.FilteredVelStatus, FilteredVort = UID.FilteredVortStatus, FilteredStrain = UID.FilteredStrainStatus)
    
    UID.OptimizationParameters()
    OptimizationStatus = UID.OptimizationStatus
    
    UID.ScalingParameters()
    ScalerMethod = UID.ScalerMethod
    
    UID.MLParameters()
    if OptimizationStatus == 'OptimizeParameters':
        import geneticAlg
        for SnapIndex in range(VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
            
            for PredictedComponent in range (0,VelMapProp.NoComponents):
                
                V_Orig.ConvertToDataFrame(ActiveFeat, VelMapProp, SnapIndex, TargetComponent=PredictedComponent, SCLMethod = ScalerMethod)
                #Prediction of velocity field for each snapshot
                #############################################################
                
                # Generating Input and Target data and split train and test portions
                # PredictedComponent=0, 1 and 2 for the x, y and z velocity components, respectively
                
                V_Orig.InputOutputSplit(TargetComponent=PredictedComponent, TestSize=UID.TestSize)
                
                #Initializing the class for Machine Learning entitled as
                # "DefiningMLMolels
                if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
                    V_Reconst = DefiningMLModels(V_Orig)
                V_Reconst.InputTrain = V_Orig.InputTrain
                V_Reconst.InputTest = V_Orig.InputTest
                V_Reconst.TargetTrain = V_Orig.TargetTrain
                V_Reconst.TargetTest = V_Orig.TargetTest
                
                if PredictedComponent==0:
                    #Generating Figure and Axes for ploting the results of optimization for SVR and MLP at a multi-axes plot
                    SP_nRow = VelMapProp.NoComponents
                    SP_nCol = 2
                    import matplotlib.pyplot as plt
                    OptFig, OptAxes = plt.subplots(SP_nRow, SP_nCol)
                
                #Optimization of SVR Parameters: Penalty coefficient and Gamma
                OptKey = 'SVR'
                KernelType = UID.SVRKernelType
                nVar = 3 # number of parameters for optimization
                # first parameter for Penalty coefficient and the second one for Gamma
                GAMinVar = np.array([10.0, 0.1, 0.001]).reshape(1, nVar)
                GAMaxVar = np.array([100.0, 40.0, 0.1]).reshape(1, nVar)
                
                OptParams = {'OptKey':OptKey, 'nVar':nVar, 'GAMinVar':GAMinVar, 'GAMaxVar':GAMaxVar, 'V_Orig':V_Orig, 'V_Reconst':V_Reconst, 'V_POD':V_POD, 'GappyProp':GappyProp, 'VelMapProp':VelMapProp, 'KernelType':KernelType, 'SnapIndex':SnapIndex, 'PredictedComponent':PredictedComponent, 'ActiveFeat':ActiveFeat, 'SVMOptCostMetric':UID.SVMOptCostMetric, 'OptAxes':OptAxes, 'UID':UID}
                SVR_Opt_Params = geneticAlg.RealGA(OptParams)
                
                
                FileName = 'SVR_Opt_Params_FOV'+str(VelMapProp.FOVIndex+1)+'.txt'
                if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent == 0:
                    file = open(FileName,'w')
                    #file = open('SVR_Opt_Params.txt','w')
                else:
                    file = open(FileName,'a')
                #file.write("{}\t{}\t{}\t{}\n".format(str(SnapIndex), str(PredictedComponent), str(SVR_Opt_Params[0]), str(SVR_Opt_Params[1])))
                file.write("{}\t{}".format(str(SnapIndex), str(PredictedComponent)))
                for i in range(0,nVar):
                    file.write("\t{}".format(str(SVR_Opt_Params[i])))
                file.write("\n")
                file.close()
                #Applying SVR on the original data based on the optimized parameters
                V_Reconst.SVRTrain(UID, Penalty=SVR_Opt_Params[0], Gamma = SVR_Opt_Params[1], Epsilon = SVR_Opt_Params[2])
                V_Reconst.SVRReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
                
                #Optimization of SVR Parameters: Penalty coefficient and Gamma
                OptKey = 'MLP'
                ActivationFunction = UID.MLPActivationFunction
                nVar = 2 # number of parameters for optimization
                # The parameters are the number of neurons in successive layers
                GAMinVar = np.array([5,5]).reshape(1, nVar)
                GAMaxVar = np.array([100, 100]).reshape(1, nVar)
                OptParams = {'OptKey':OptKey, 'nVar':nVar, 'GAMinVar':GAMinVar, 'GAMaxVar':GAMaxVar, 'V_Orig':V_Orig, 'V_Reconst':V_Reconst, 'GappyProp':GappyProp, 'VelMapProp':VelMapProp, 'ActivFunc':ActivationFunction, 'SnapIndex':SnapIndex, 'PredictedComponent':PredictedComponent, 'ActiveFeat':ActiveFeat, 'MLPOptCostMetric':UID.MLPOptCostMetric, 'OptAxes':OptAxes, 'UID':UID}
                MLP_Opt_Params = geneticAlg.RealGA(OptParams)
                
                FileName = 'MLP_Opt_Params_FOV'+str(VelMapProp.FOVIndex+1)+'.txt'
                if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent == 0:
                    file = open(FileName,'w')
                else:
                    file = open(FileName,'a')
                file.write("{}\t{}".format(str(SnapIndex), str(PredictedComponent)))
                for i in range(0,nVar):
                    file.write("\t{}".format(str(MLP_Opt_Params[i])))
                file.write("\n")
                file.close()
                
                #Applying MLP on the original data based on the optimized parameters
                if MLP_Opt_Params.size == 1:
                    HLSize = (int(MLP_Opt_Params[0]),)
                else:
                    HLSize = []
                    for i in range(0,MLP_Opt_Params.size):
                        HLSize.append(int(MLP_Opt_Params[i]))
                    HLSize = tuple(HLSize)
                V_Reconst.MLPTrain(UID, HidLayerSize = HLSize, ActivFunc = ActivationFunction)
                V_Reconst.MLPReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
    elif OptimizationStatus == 'UseOptimizationFiles':
        #Detecting number of lines
        SVMFileName = 'SVR_Opt_Params_FOV'+str(VelMapProp.FOVIndex+1)+'.txt'
        MLPFileName = 'MLP_Opt_Params_FOV'+str(VelMapProp.FOVIndex+1)+'.txt'
        num_lines = sum(1 for line in open(SVMFileName))
        
        SVMFile = open(SVMFileName,'r')
        MLPFile = open(MLPFileName,'r')
        for iteration in range(0, num_lines):
            print('iteration: {}'.format(iteration))
            SVM_line_info = SVMFile.readline().split()
            MLP_line_info = MLPFile.readline().split()
            SnapIndex = int(SVM_line_info[0])
            PredictedComponent = int(SVM_line_info[1])
            
            if SnapIndex >= VelMapProp.StartSnapIndex-1 and SnapIndex <= VelMapProp.EndSnapIndex-1:
                V_Orig.ConvertToDataFrame(ActiveFeat, VelMapProp, SnapIndex, TargetComponent=PredictedComponent, SCLMethod = ScalerMethod)
                V_Orig.InputOutputSplit(TargetComponent=PredictedComponent, TestSize=UID.TestSize)
                
                #Initializing the class for Machine Learning entitled as
                # "DefiningMLMolels
                if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
                    V_Reconst = DefiningMLModels(V_Orig)
                V_Reconst.InputTrain = V_Orig.InputTrain
                V_Reconst.InputTest = V_Orig.InputTest
                V_Reconst.TargetTrain = V_Orig.TargetTrain
                V_Reconst.TargetTest = V_Orig.TargetTest
                
                #Applying SVR on the original data based on the optimized parameters
                OptPenalty = float(SVM_line_info[2])
                OptGamma = float(SVM_line_info[3])
                OptEpsilon = float(SVM_line_info[4])
                print(OptPenalty, OptGamma, OptEpsilon)
                V_Reconst.SVRTrain(UID, Penalty=OptPenalty, Gamma = OptGamma, Epsilon=OptEpsilon)
                V_Reconst.SVRReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
            
                #Applying MLP Neural Network on the original data based on the optimized parameters
                MLP_Opt_Params = []
                counter = 2
                MLP_Layer_Size = np.array(MLP_line_info).size - 2
                for Layer in range(0, MLP_Layer_Size):
                    MLP_Opt_Params.append(float(MLP_line_info[counter]))
                    counter += 1
                
                if MLP_Layer_Size == 1:
                    HLSize = (int(MLP_Opt_Params[0]),)
                else:
                    HLSize = []
                    for i in range(0,np.array(MLP_Opt_Params).size):
                        HLSize.append(int(MLP_Opt_Params[i]))
                    HLSize = tuple(HLSize)
                print(HLSize)
                V_Reconst.MLPTrain(UID, HidLayerSize = HLSize, ActivFunc = ActivationFunction)
                V_Reconst.MLPReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
        
        SVMFile.close()
        MLPFile.close()
    
    elif OptimizationStatus == 'CustomizeParameters':
        for SnapIndex in range(VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
            if SnapIndex in UID.SnapshotsForNonGPODRegressors:
            #for SnapIndex in range(VelMapProp.StartSnapIndex-1, 8):
                        
                for PredictedComponent in range (0,VelMapProp.NoComponents):
                    V_Orig.ConvertToDataFrame(ActiveFeat, VelMapProp, SnapIndex, TargetComponent=PredictedComponent, SCLMethod = ScalerMethod)
                    V_Orig.InputOutputSplit(TargetComponent=PredictedComponent, TestSize=UID.TestSize)
                    #Initializing the class for Machine Learning entitled as
                    # "DefiningMLMolels
                    if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
                        V_Reconst = DefiningMLModels(V_Orig)
                    V_Reconst.InputTrain = V_Orig.InputTrain
                    V_Reconst.InputTest = V_Orig.InputTest
                    V_Reconst.TargetTrain = V_Orig.TargetTrain
                    V_Reconst.TargetTest = V_Orig.TargetTest
                    
                    #Applying SVR on the original data based on the customized parameters
                    CustomizedPenalty = UID.CustomizedPenalty[PredictedComponent]
                    CustomizedGamma = UID.CustomizedGamma[PredictedComponent]
                    CustomizedEpsilon = UID.CustomizedEpsilon[PredictedComponent]
                
                    V_Reconst.SVRTrain(UID, Penalty=CustomizedPenalty, Gamma = CustomizedGamma, Epsilon=CustomizedEpsilon)
                    V_Reconst.SVRReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
                    V_Reconst.SVR_CV_Score(UID, Penalty = CustomizedPenalty, Gamma = CustomizedGamma, Epsilon=CustomizedEpsilon)
                    
                    CompletionPercentage = (SnapIndex*VelMapProp.NoComponents+PredictedComponent+1)/(VelMapProp.NoSnapshots*VelMapProp.NoComponents)*100
                    print('______________________________________________________________')
                    print('{0:6.2f}% Completed. SVR Reconstruction: C={1:8.4f}, Gamma={2:8.4f}, Epsilon={3:9.5f}'.format(CompletionPercentage, CustomizedPenalty, CustomizedGamma, CustomizedEpsilon))
                    print('SVR Training Score: {0:9.5f}\nSVR CV Score Mean/StdDV: {1:9.5f}/{2:9.5f}\nSVR Test Score:{3:9.5f}'.format(V_Reconst.SVRTrainingScore, V_Reconst.SVR_CVScore_mean, V_Reconst.SVR_CVScore_StdDV, V_Reconst.SVRPredictionScore))
                    
                    #Applying MLP Neural Network on the original data based on the optimized parameters
                    CustomizedMLPLayersConfig = UID.CustomizedLayersConfig[PredictedComponent]
                    ActivationFunction = UID.MLPActivationFunction
                    MLP_Layer_Size = np.array(CustomizedMLPLayersConfig).size
                    if MLP_Layer_Size == 1:
                        HLSize = (int(CustomizedMLPLayersConfig[0]),)
                    else:
                        HLSize = []
                        for i in range(0,np.array(CustomizedMLPLayersConfig).size):
                            HLSize.append(int(CustomizedMLPLayersConfig[i]))
                        HLSize = tuple(HLSize)
                    
                    V_Reconst.MLPTrain(UID, HidLayerSize = HLSize, ActivFunc = ActivationFunction)
                    V_Reconst.MLPReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
                    V_Reconst.MLP_CV_Score(UID, HidLayerSize = HLSize, ActivFunc = ActivationFunction)
                    
                    print('MLP Reconstruction: Hidden Layer Size = {}'.format(HLSize))
                    print('MLP Training Score: {0:9.5f}\nMLP CV Score Mean/StdDV: {1:9.5f}/{2:9.5f}\nMLP Test Score:{3:9.5f}'.format(V_Reconst.MLPTrainingScore,V_Reconst.MLP_CVScore_mean, V_Reconst.MLP_CVScore_StdDV, V_Reconst.MLPPredictionScore))
                
    elif OptimizationStatus == 'CrossValidation':
        CVResultsSheet = XLSXResults.add_worksheet("CVResults")
        XLSRow = 0
        XLSCVTitles = ['Snapshot Index', 'Velocity Component', 'Predictive Model', 'Best Hyperparameters']
        for col in range(0,len(XLSCVTitles)):
            CVResultsSheet.write(XLSRow, col, XLSCVTitles[col])
            
        for SnapIndex in range(VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
            if SnapIndex in UID.SnapshotsForNonGPODRegressors:
            #for SnapIndex in range(VelMapProp.StartSnapIndex-1, 8):
                        
                for PredictedComponent in range (0,VelMapProp.NoComponents):
                    V_Orig.ConvertToDataFrame(ActiveFeat, VelMapProp, SnapIndex, TargetComponent=PredictedComponent, SCLMethod = ScalerMethod)
                    V_Orig.InputOutputSplit(TargetComponent=PredictedComponent, TestSize=UID.TestSize)
                    #Initializing the class for Machine Learning entitled as
                    # "DefiningMLMolels
                    if SnapIndex == VelMapProp.StartSnapIndex-1 and PredictedComponent==0:
                        V_Reconst = DefiningMLModels(V_Orig)
                    V_Reconst.InputTrain = V_Orig.InputTrain
                    V_Reconst.InputTest = V_Orig.InputTest
                    V_Reconst.TargetTrain = V_Orig.TargetTrain
                    V_Reconst.TargetTest = V_Orig.TargetTest
                    
                    #Applying SVR on the original data based on the GridSearchCV parameters
                    CVParams = {'C':UID.PenaltyRange, 'gamma':UID.GammaRange, 'epsilon':UID.EpsilonRange, 'kernel':UID.Kernels}
                    V_Reconst.SVR_CV_Train(UID, CVParams)
                    V_Reconst.SVRReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
                    XLSRow+=1
                    XLSRowData = [SnapIndex+1, PredictedComponent+1, 'SVR', V_Reconst.svr.best_params_['C'], V_Reconst.svr.best_params_['gamma'], V_Reconst.svr.best_params_['epsilon'], V_Reconst.svr.best_params_['kernel']]
                    nCols = 7
                    for col in range(0, nCols):
                        CVResultsSheet.write(XLSRow, col, XLSRowData[col])
                    
                    #Applying MLP on the original data based on the GridSearchCV parameters
                    CVParams = {'activation':UID.ActivationFunctions, 'hidden_layer_sizes':UID.HLSize}
                    V_Reconst.MLP_CV_Train(UID, CVParams)
                    V_Reconst.MLPReconstruct(ActiveFeat, V_Orig, VelMapProp, SnapIndex, PredictedComponent, SCLMethod = ScalerMethod)
                    XLSRow+=1
                    XLSRowData = [SnapIndex+1, PredictedComponent+1, 'MLP', V_Reconst.mlp.best_params_['activation']]
                    XLSRowData.extend(list(V_Reconst.mlp.best_params_['hidden_layer_sizes']))
                    nCols = 4+len(V_Reconst.mlp.best_params_['hidden_layer_sizes'])
                    for col in range(0,nCols):
                        CVResultsSheet.write(XLSRow, col, XLSRowData[col])
                
    # Masking the Predicted Velocity field for the rotor region
    if UID.VelMaskStatus==True and (UID.FOVIndex==1 or UID.FOVIndex==2):
        V_Reconst.VelocityMask(VelMapProp, UID)
    
    from subplot_configs import Config1
    #Config1(V_Orig.VOrig[VelMapProp.InstSnapIndex,:,:,PredictedComponent-1], V_Reconst.V_predicted_SVR[VelMapProp.InstSnapIndex,:,:,PredictedComponent-1], VelMapProp, FigureSize=(10,7), N_Rows = 2, N_Cols = 1)
    #Config1(V_Orig.VOrig[VelMapProp.InstSnapIndex,:,:,PredictedComponent-1], V_Reconst.V_predicted_MLP[VelMapProp.InstSnapIndex,:,:,PredictedComponent-1], VelMapProp, FigureSize=(10,7), N_Rows = 2, N_Cols = 1)
    
    #Specifying the target quantities for plot generations
    VReconst = {'SVR':V_Reconst.V_predicted_SVR,
                'MLP':V_Reconst.V_predicted_MLP,
                'GPOD':V_POD.UPOD,
                'NNI':V_Interpol.VNearestND,
                'Cubic':V_Interpol.VGridCubic,
                'TLI':V_Interpol.VGridLinear,
                'MMI':V_Interpol.VMovingMedian,
                'GMAI':V_Interpol.VMovingAveGaussian,
                'UMAI':V_Interpol.VMovingAveUniform}
                #'Nearest-Neighbor':V_Interpol.VGridNearestNeigh,
    SelectedRegressors = ['SVR','MLP','GPOD','NNI'
                          ,'TLI','MMI'
                          ,'GMAI','UMAI']
    #Config1(V_Orig.VOrig[VelMapProp.InstSnapIndex,:,:,PredictedComponent-1], V_POD.UPOD[VelMapProp.InstSnapIndex,:,:,PredictedComponent-1], VelMapProp, FigureSize=(10,7), N_Rows = 2, N_Cols = 1)
    
    from subplot_configs import Config2, Config3, config4, OutputTargetComparisonByScatter
    from statistical_evaluation import StatEvaluation
    StatResults = StatEvaluation(V_Orig, VReconst, SelectedRegressors, GappyProp, VelMapProp)
    UID.PlotsInputParameters()
    SnapshotForPresentation = UID.SnapForPresentation
    ModelEvalWorkSheet = XLSXResults.add_worksheet("ModelsEvaluation")
    
    for SnapIndex in range (VelMapProp.StartSnapIndex-1, VelMapProp.EndSnapIndex):
        if SnapIndex in SnapshotForPresentation:
            #Model evaluation based on continuity
            CBE = ContinuityBasedEvaluation(V_Orig.VOrigDuplicate[SnapIndex,:,:,:], VReconst, SelectedRegressors, GappyProp.LocationsForEvaluation[SnapIndex,:,:,0], VelMapProp, FOVIndex, SnapIndex)
            CBE.LocationsForContinuityEvals()
            CBE.ContinuityMetrics(SnapIndex, SelectedRegressors, UID.PIV_XSpacing, UID.PIV_YSpacing, UID)
            CBE.PredictionErrors(SelectedRegressors)
            
            ContAbsDeviation['FOV'+str(FOVIndex)] = {}
            for rgrs in SelectedRegressors:
                ContAbsDeviation['FOV'+str(FOVIndex)][rgrs] = CBE.ContAbsDeviation[rgrs]

            
            ContMetric['FOV'+str(FOVIndex)] = {}
            ContMetric['FOV'+str(FOVIndex)]['Orig'] = CBE.ContMetricArray['Orig']
            for rgrs in SelectedRegressors:
                ContMetric['FOV'+str(FOVIndex)][rgrs] = CBE.ContMetricArray[rgrs]
        
        
        for component in range (0, VelMapProp.NoComponents):
            if SnapIndex in SnapshotForPresentation:
                nCountors = UID.nCountors
                OutierPercentageForLevels=UID.OutierPercentageForLevels
                if UID.ContourLevelingStatus == 'Customized':
                    MaxLevel = UID.MaxLevel[component]
                elif UID.ContourLevelingStatus == 'Automatic':
                    MaxLevel = np.max(np.array([np.percentile(V_Orig.VOrigDuplicate[SnapIndex,:,:,component],OutierPercentageForLevels), np.percentile(V_Orig.VOrig[SnapIndex,:,:,component],OutierPercentageForLevels), np.percentile(V_Reconst.V_predicted_SVR[SnapIndex,:,:,component],OutierPercentageForLevels), np.percentile(V_Reconst.V_predicted_MLP[SnapIndex,:,:,component],OutierPercentageForLevels), np.percentile(V_POD.UPOD[SnapIndex,:,:,component],OutierPercentageForLevels)]))
                TempVec1 = np.copy(V_Orig.VOrigDuplicate[SnapIndex,:,:,component])
                TempVec2 = np.copy(V_Orig.VOrig[SnapIndex,:,:,component])
                TempVec3 = np.copy(V_Reconst.V_predicted_SVR[SnapIndex,:,:,component])
                TempVec4 = np.copy(V_Reconst.V_predicted_MLP[SnapIndex,:,:,component])
                TempVec5 = np.copy(V_POD.UPOD[SnapIndex,:,:,component])
                np.place(TempVec1,TempVec1==0,np.Inf)
                np.place(TempVec2,TempVec2==0,np.Inf)
                np.place(TempVec3,TempVec3==0,np.Inf)
                np.place(TempVec4,TempVec4==0,np.Inf)
                np.place(TempVec5,TempVec5==0,np.Inf)
                #MinLabel = np.min(np.array([np.min(TempVec1),np.min(TempVec2),np.min(TempVec3),np.min(TempVec4),np.min(TempVec5)]))
                if UID.ContourLevelingStatus == 'Customized':
                    MinLevel = UID.MinLevel[component]
                elif UID.ContourLevelingStatus == 'Automatic':
                    MinLevel = np.min(np.array([np.percentile(TempVec1,100-OutierPercentageForLevels),np.percentile(TempVec2,100-OutierPercentageForLevels),np.percentile(TempVec3,100-OutierPercentageForLevels),np.percentile(TempVec4,100-OutierPercentageForLevels),np.percentile(TempVec5,100-OutierPercentageForLevels)]))
                ContourLevels = list(np.arange(MinLevel,MaxLevel,(MaxLevel-MinLevel)/nCountors))
                #Config2(SnapIndex, component, V_Orig.VOrigDuplicate[SnapIndex,:,:,component], V_Orig.VOrig[SnapIndex,:,:,component], V_Reconst.V_predicted_SVR[SnapIndex,:,:,component], V_Reconst.V_predicted_MLP[SnapIndex,:,:,component], V_POD.UPOD[SnapIndex,:,:,component], VelMapProp, FigureSize=(10,7), N_Rows = 2, N_Cols = 3, ContourLevels = ContourLevels)
                Config2(UID, SnapIndex, component, TempVec1, TempVec2, TempVec3, TempVec4, TempVec5, VelMapProp, FigureSize=UID.FigSizeContours, N_Rows = UID.NoContourPlots, N_Cols = 1, ContourLevels = ContourLevels)
            StatResults.ElementaryEvaluation(SelectedRegressors, UID, SnapIndex, EvaluatingComponent = component)
            if SnapIndex == UID.SnapIndexForBoxPlot:
                if component == 0:
                    for rgrs in SelectedRegressors:
                        multiplier=1
                        if rgrs not in ['SVR', 'MLP']:
                            multiplier=2
                        PredDevForBoxPlot_C1.append(multiplier * StatResults.PredDeviation[rgrs].flatten())

                elif component == 1:
                    for rgrs in SelectedRegressors:
                        multiplier=1
                        if rgrs not in ['SVR', 'MLP']:
                            multiplier=2
                        PredDevForBoxPlot_C2.append(multiplier * StatResults.PredDeviation[rgrs].flatten())

                elif component == 2:
                    for rgrs in SelectedRegressors:
                        multiplier=1
                        if rgrs not in ['SVR', 'MLP']:
                            multiplier=2
                        PredDevForBoxPlot_C3.append(multiplier * StatResults.PredDeviation[rgrs].flatten())

                
            StatResults.Results(SelectedRegressors, SnapIndex, component, ModelEvalWorkSheet, VelMapProp)
            if SnapIndex in SnapshotForPresentation:
                OutputTargetComparisonByScatter(UID, SelectedRegressors, StatResults, SnapIndex, component, N_Rows=3, N_Cols = 3, FigureSize=UID.FigSizeScatters)
                QuantityLabels = ['Original Data','Gap-augmented Data','SVR Prediction','MLP Prediction','GPOD Prediction']
                Config3(UID, SnapIndex, component, StatResults, V_Orig.VOrigDuplicate[SnapIndex,UID.HLinePosition,:,component], V_Orig.VOrig[SnapIndex,UID.HLinePosition,:,component], V_Reconst.V_predicted_SVR[SnapIndex,UID.HLinePosition,:,component], V_Reconst.V_predicted_MLP[SnapIndex,UID.HLinePosition,:,component], V_POD.UPOD[SnapIndex,UID.HLinePosition,:,component], QuantityLabels, VelMapProp, FigureSize=UID.FigSizeXYPlots, N_Rows = UID.NoXYPlots, N_Cols = 1)
                from plots import GeneratePlots
                for rgrs in SelectedRegressors:
                    Fig, ax = plt.subplots(1,1)
                    Figure = GeneratePlots(Fig, ax)
                    FigureSize = (6,6)
                    MNAE = StatResults.PredMNAE[rgrs]
                    RMSNAE = StatResults.PredRMSNAE[rgrs]
                    MedNAE = StatResults.PredMedNAE[rgrs]
                    R2_Score = StatResults.PredR2_Score[rgrs]
                    AxTitle = "{0}\nMNAE:{1:.2f},   RMSNAE:{2:.2f}\nMedNAE:{3:.2f},   ".format(rgrs, MNAE, RMSNAE, MedNAE) + r'$R_2-score:$'+' {0:.3f}'.format(R2_Score)
                    Figure.OutputTargetScatterPlot(StatResults.Output[rgrs], StatResults.Target, XLabel = r'{} Prediction: $U_{}^{}$ (m/s)'.format(rgrs, component+1, SnapIndex+1), YLabel = r'True Value: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize, COLOR = 'b', ALPHA = 0.4, Slope45=True, LabelSize=14, TitleSize=14, TicLabelSize=12, QuantityLabel=rgrs, LegendStatus='Off')
                    Fig.tight_layout()
        if SnapIndex in SnapshotForPresentation:
            TempVec0 = {'x': np.copy(V_Orig.VOrig[SnapIndex,:,:,0]), 'y': np.copy(V_Orig.VOrig[SnapIndex,:,:,1])}
            TempVec1 = {'x': np.copy(V_Orig.VOrigDuplicate[SnapIndex,:,:,0]), 'y': np.copy(V_Orig.VOrigDuplicate[SnapIndex,:,:,1])}
            TempVec2 = {'x': np.copy(V_Reconst.V_predicted_SVR[SnapIndex,:,:,0]), 'y': np.copy(V_Reconst.V_predicted_SVR[SnapIndex,:,:,1])}
            TempVec3 = {'x': np.copy(V_Reconst.V_predicted_MLP[SnapIndex,:,:,0]), 'y': np.copy(V_Reconst.V_predicted_MLP[SnapIndex,:,:,1])}
            TempVec4 = {'x': np.copy(V_POD.UPOD[SnapIndex,:,:,0]), 'y': np.copy(V_POD.UPOD[SnapIndex,:,:,1])}
            config4(UID, TempVec0, TempVec1, TempVec2, TempVec3, TempVec4, VelMapProp, 1, 3, FigureSize=UID.FigSizeQuiver)

    UID.XLSX_Input_Params(XLSXResults)

    XLSXResults.close()

# Generating an object for the input and output vector-maps for presenting the results
#from main_results import MainInputsOutputs


from boxPlots import BoxPlot, BoxPlot2

DataForBoxPlot = {'1':PredDevForBoxPlot_C1, '2':PredDevForBoxPlot_C2, '3':PredDevForBoxPlot_C3}

BoxPlot(DataForBoxPlot, SelectedRegressors, FOVS, UID)

#ploting the errors box plots based on the continuity metric
DataForBoxPlot2 = []
from plots import GeneratePlots
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from math import sqrt

for FOVIndex in FOVS:
    for rgrs in SelectedRegressors:
        if rgrs not in ['SVR', 'MLP']:
            DataForBoxPlot2.append(2*ContAbsDeviation['FOV'+str(FOVIndex)][rgrs]/(ContMetric['FOV'+str(FOVIndex)]['Orig'].std()))
        else:
            DataForBoxPlot2.append(ContAbsDeviation['FOV'+str(FOVIndex)][rgrs]/(ContMetric['FOV'+str(FOVIndex)]['Orig'].std()))

    Fig, ax = plt.subplots(1,1)
    Figure = GeneratePlots(Fig, ax)
    AxTitle = 'The code below should be modified'

    FigureSize = (6,6)
    CBE_ScatterPlotsRegressors = ['SVR', 'MLP', 'MMI']
    Colors = ['y', 'g', 'c']
    COUNTER=0
    for rgrs in CBE_ScatterPlotsRegressors:
        Figure.OutputTargetScatterPlot(ContMetric['FOV'+str(FOVIndex)][rgrs], ContMetric['FOV'+str(FOVIndex)]['Orig'], XLabel = r'Predicted $\mathit{MC}$ (1/s)', YLabel = r'Original $\mathit{MC}$ (1/s)', AxesTitle = AxTitle, FigSize=FigureSize, COLOR = Colors[COUNTER], ALPHA = 0.6, Slope45=False, LabelSize=14, TitleSize=14, TicLabelSize=12, QuantityLabel=rgrs)
        COUNTER+=1

    
    for rgrs in SelectedRegressors:
        Fig, ax = plt.subplots(1,1)
        Figure = GeneratePlots(Fig, ax)
        FigureSize = (6,6)
        MAE = mean_absolute_error(ContMetric['FOV'+str(FOVIndex)]['Orig'], ContMetric['FOV'+str(FOVIndex)][rgrs])
        RMSE = sqrt(mean_squared_error(ContMetric['FOV'+str(FOVIndex)]['Orig'], ContMetric['FOV'+str(FOVIndex)][rgrs]))
        MedAE = median_absolute_error(ContMetric['FOV'+str(FOVIndex)]['Orig'], ContMetric['FOV'+str(FOVIndex)][rgrs])
        R2_Score = r2_score(ContMetric['FOV'+str(FOVIndex)]['Orig'], ContMetric['FOV'+str(FOVIndex)][rgrs])
        AxTitle = "{0}\nMAE:{1:.2f} 1/s,   RMSE:{2:.2f} 1/s\nMedAE:{3:.2f} 1/s,   ".format(rgrs, MAE, RMSE, MedAE) + r'$R_2-score:$'+' {0:.3f}'.format(R2_Score)
        Figure.OutputTargetScatterPlot(ContMetric['FOV'+str(FOVIndex)][rgrs], ContMetric['FOV'+str(FOVIndex)]['Orig'], XLabel = r'Predicted $\mathit{MC}$ (1/s)', YLabel = r'Original $\mathit{MC}$ (1/s)', AxesTitle = AxTitle, FigSize=FigureSize, COLOR = 'b', ALPHA = 0.4, Slope45=True, LabelSize=14, TitleSize=14, TicLabelSize=12, QuantityLabel=rgrs, PlotRange = 'Manual', LegendStatus='Off')
        Fig.tight_layout()
        
        
        
    
    
BoxPlot2(DataForBoxPlot2, SelectedRegressors, FOVS, UID)


EndTime = time()
ExecutionTime = EndTime - StartTime
print("___________________________________________")
print("Execution Time: {} sec".format(ExecutionTime))