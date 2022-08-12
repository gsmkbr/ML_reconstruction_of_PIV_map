class UserInputData:
    def __init__(self):
        pass
    
    def VelocityInputData(self, FOVIndex):
        #The FOV number for analysis
        self.FOVIndex = FOVIndex
        # Number of total FOVs
        self.N_FOVs = 4
        # Instantaneous snapshot index for analysis of one selected snapshot
        self.InstSnapIndex = 1
        # Number of velocity components
        self.NoComponents = 3
        # Starting index of snapshots for analysis
        self.StartSnapIndex = 1
        # Ending indes of snapshots for analysis
        self.EndSnapIndex = 5
        
        self.SnapshotsForNonGPODRegressors = [0]
        
    def VelMaskProperties(self):
        # The required data for masking the rotor region
        # Activating/DeActivating the masking operation (True or False)
        self.VelMaskStatus = True
        # Rotor radius in the unit of FOV spacing index
        self.RotorRadius = 169
        # Vertical distance between the coordinate reference point and the rotor center in the unit of FOV spacing index
        self.RefToCenterYDist = 143
        
    def GappyProperties(self, VelMapProp):
        # minimum and maximum size of artificial gapped area
        self.GapSize = [2, 6]
        # the percentage of required gappiness in each snapshot
        # For FOVs with more initial gappiness, larger velue of artificial gappiness percentage is required
        # the following calculations are done to make the amount of gappiness in various FOVs the same
        self.MaskPercent = 23.5  #percentage of gappiness due to the rotor in FOVs 1 and 2
        
        self.GappinessPercentageWithoutRotor = 50 # the amount of gappiness in out-rotor region
        
        NRows = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        NCols = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        if self.FOVIndex == 1 or self.FOVIndex == 2:
            UnderRotorPoints = self.MaskPercent/100*NRows*NCols
            OuterRotorPoints = NRows*NCols-UnderRotorPoints
            OuterRotorMaskedPoints = self.GappinessPercentageWithoutRotor/100*OuterRotorPoints
            TotalMaskedPoints = UnderRotorPoints + OuterRotorMaskedPoints
            self.GappinessPercent = TotalMaskedPoints/NRows/NCols*100
        elif self.FOVIndex == 3 or self.FOVIndex == 4:
            self.GappinessPercent = self.GappinessPercentageWithoutRotor
            
        
    def VelocityOperationsProp(self):
        # defining multiple filter length scales in x-direction as a list IN AN ASCENDING ORDER
        self.FiltLenX = [7, 9]
        # defining multiple filter length scales in y-direction as a list
        self.FiltLenY = [7, 9]
        # defining the filter type as an string ('Gaussian' or 'BoxUniform')
        self.FiltType = 'Gaussian'
        # defining the PIV grid spacing in x and y directions in milimeters
        self.PIV_XSpacing = 0.001
        self.PIV_YSpacing = 0.001
        
    def GappyPODProperties(self):
        #number of employed modes in Gappy POD algorithms
        self.nMode = 2
        # Maximum number of iterations in the main GPOD algorithm loop
        self.MaxIter = 10
        
    def ActiveFeaturesConfig(self):
        # defining the status of each predictor feature to be active or passive (True or False)
        self.PositionStatus = True
        self.FilteredVelStatus = True
        self.FilteredVortStatus = False
        self.FilteredStrainStatus = False
        
    def OptimizationParameters(self):
        # difining the optimization configuration:
            # 'CustomizeParameters' for a specific configuration of parameters
            # 'OptimizeParameters' for optimizatin the hyperparameters
            # 'UseOptimizationFiles' for utilizating the optimization result files, prepared beforehand
            # 'CrossValidation' for k-fold Cross Validation
        self.OptimizationStatus = 'CustomizeParameters'
        
        # Specifying the method of calculating the models score in optimization
            # 'CrossValidation' for k-fold cross validation scoring based on validation data
            # 'ClassicPrediction' for scoring of classic prediction based on test data
        self.OptimizationScoringMethod = 'CrossValidation'
        
        #SVM Optimization metric
        # 'MNAE': Mean Normalized Absolute Error
        # 'RMSNAE': Root Mean Square of Normalized Absolute Error
        # 'MedNAE': Median of Normalized Absolute Error
        # 'R2_Score': R2-score
        self.SVMOptCostMetric = 'R2_Score'
        self.MLPOptCostMetric = self.SVMOptCostMetric
        
    def ScalingParameters(self):
        # difining the scaling method:
            #'None': without scaling
            #'StandardScaler': Using StandardScaler approach
            #'MinMaxScaler': Using MinMaxScaler approach
            #'RobustScaler': Using RobustScaler approach
            #'Normalizer': Using normalize approach
        self.ScalerMethod = 'StandardScaler'
        
    def MLParameters(self):
        # difining the test size in train-test split of various learning algorithms
        self.TestSize = 0.001
        #_________________________________________________
        # Defining Properties of SVR algorithm 
        # The SVR kernel type: {'linear', 'poly', 'rbf', 'sigmoid'}
        self.SVRKernelType = 'rbf'
        # defining the customized hyperparameters for SVR:
        self.CustomizedPenalty = [48.78,48.15,28.99]  #50.0
        self.CustomizedGamma = [16.89,8.72,4.5]    #1.0
        self.CustomizedEpsilon = [0.003, 0.014, 0.038]  #0.01
        self.CustomKFoldCVNo = 3
        #_________________________________________________
        # Defining Properties of MLP algorithm
        # The MLP Activation Function: {'identity', 'logistic', 'tanh', 'relu'}
        self.MLPActivationFunction = 'relu'
        # The solver for weight optimization ('lbfgs', 'sgd', 'adam')
        self.Solver = 'lbfgs'
        # learning_rate ('constant', 'invscaling', 'adaptive')
        self.LearnRateStatus = 'adaptive'
        # The initial learning rate used (Only used when solver=’sgd’ or ‘adam’)
        self.LearnRateInit = 0.001
        # The exponent for inverse scaling learning rate (Only used when solver=’sgd’ and learning_rate is set to ‘invscaling’)
        self.PowerT = 0.5
        # number of iterations in learning algorithm
        self.MLPMaxIter = 100
        # Tolerance for the optimization. When the loss or score is not improving by at least 'tol' for 'n_iter_no_change' consecutive iterations, convergence is considered to be reached and training stops.
        self.ValidationTol = 1e-2
        # Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.
        self.EarlyStopStatus = True
        # The proportion of training data to set aside as validation set for early stopping. terminate training when validation score is not improving by at least tol for two consecutive epochs. Must be between 0 and 1.
        self.ValidFraction = 0.1
        # Momentum for gradient descent update (Only used when solver=’sgd’). Should be between 0 and 1.
        self.MomentumCoeff = 0.9
        
        # defining the customized hyperparameters for MLP:
        self.CustomizedLayersConfig = [[75,75], [75,75], [71,93]] #[[47,71], [18,71], [65,75]]   #(20,20)
        
        #Cross Validation Status: ('GridSearchCV' or 'RandomizedSearchCV')
        self.CVStatus = 'RandomizedSearchCV'
        
        #Defining a function to generate user-defined number of MLP layers config (number of layers and number of neurons in each layer)
        def MLPLayerConfig(GenerationConfig, nNeuronsRange):
            from random import randint
            nNeuronsMin = nNeuronsRange[0]
            nNeuronsMax = nNeuronsRange[1]
            LayersConfig = []
            for Config in GenerationConfig:
                nLayers = Config[0]
                nConfigs = Config[1]
                for config in range(0, nConfigs):
                    TempList = []
                    for layer in range(0,nLayers):
                        nNeuronsLocal = randint(nNeuronsMin, nNeuronsMax)
                        TempList.append(nNeuronsLocal)
                    LayersConfig.append(tuple(TempList))
            return LayersConfig
            
        # Cross Validation parameters
            # for SVR:
        import numpy as np
        if self.CVStatus == 'GridSearchCV':
            nRandSamples = 3
            self.PenaltyRange = np.logspace(-3, 2, num=nRandSamples)
            self.GammaRange = np.logspace(-3, 2, num=nRandSamples)
            self.EpsilonRange = np.logspace(-3, 0, num=nRandSamples)
            self.Kernels = ('rbf', 'linear', 'sigmoid')
                #for MLP:
            # Difining Hidden Layers config as a list of tuples;
            # Each tuple includes multiple MLPs with the same number of layers
            # first number in each tuple defines the number of layers
            # and second number in each tuple defines the number of randomly generated MLPs with randon number of neurons in each layer
            self.HiddenHayersConfig = [(1,3), (2,4), (3,5)]
            #Difining the minimum and maximum number of randomly generated neorons
            nNeuronsRange = [5,50]
            self.HLSize = MLPLayerConfig(self.HiddenHayersConfig, nNeuronsRange)
            self.ActivationFunctions = ('identity', 'logistic', 'tanh', 'relu')
            
        elif self.CVStatus == 'RandomizedSearchCV':
            
            #Number of parallel tasks in Randomized search CV
            self.No_Jobs_RSCV = 6
            #for SVR:
            #number of iterations in Randomized search CV algorithm for SVR Algorithm
            self.RSCV_SVR_nIter = 200
            #generating exponential distributions for Penalty coefficient and gamma coefficient
            nRandSamples = 1000
            self.PenaltyRange = np.logspace(0, 2, num=nRandSamples)
            self.GammaRange = np.logspace(-1, 2, num=nRandSamples)
            self.EpsilonRange = np.logspace(-3, -1, num=nRandSamples)
            self.Kernels = ('rbf',)
            #for MLP:
            #number of iterations in Randomized search CV algorithm for MLP Algorithm
            self.RSCV_MLP_nIter = 50
            # Difining Hidden Layers config as a list of tuples;
            # Each tuple includes multiple MLPs with the same number of layers
            # first number in each tuple defines the number of layers
            # and second number in each tuple defines the number of randomly generated MLPs with randon number of neurons in each layer
            self.HiddenHayersConfig = [(2,500)]
            #Difining the minimum and maximum number of randomly generated neorons
            nNeuronsRange = [30,85]
            self.HLSize = MLPLayerConfig(self.HiddenHayersConfig, nNeuronsRange)
            self.ActivationFunctions = ('relu',)
            
        # number of folds in k-fold cross validation
        self.nCV = 3
        
        # Defining the type of error evaluation
        # 'Absolute' for the absolute error between prediction and target
        # 'Normalized' for the relative error between prediction and target
        self.PredicitonErrorType = 'Normalized'
        
    def PlotsInputParameters(self):
        # difining specific snapshots for which the graphs are presented (as a list of snapshot indices)
        self.SnapForPresentation = [0]
        # Number of levels in each contour plot
        self.nCountors = 20
        # Position of horizontal line for evaluating the performance of predictions as 1D plots
        self.HLinePosition = 40
        self.DrawHLine = True
        #difining the contour leveling status ('Customized' or 'Automatic')
        self.ContourLevelingStatus = 'Customized'
        # difining a percentile threshold for contour plots to avoid the outliers effect
        self.OutierPercentageForLevels=99.99
        # difining a customized range for three velocity components contour levels
        self.MinLevel = [10, -6, -7]
        self.MaxLevel = [30, 22, 12]
        # defining the colormap for contour plots
        self.CMAP = 'inferno_r'
        #self.CMAP = 'plasma_r'
        #self.CMAP = 'viridis_r'
        # Specifying to insert color bar, axes labels, axes titles
        self.CMAPStatus = False
        self.AxLabelStatus = False
        self.AxTitleStatus = False
        # defining the ploting status of the ML models
        self.MLPlotStatus = {'SVM':True, 'MLP':True, 'GPOD': True}
        #defining the number of active contour plots
        self.NoContourPlots = 5
        #defining the number of active scatter plots
        self.NoScatterPlots = 3
        #defining the number of active 1-D plots
        self.NoXYPlots = 3
        # defining the size of figures
            # for contour plots
        self.FigSizeContours = (4,12)
            # for vector plots
        self.FigSizeQuiver = (12, 6)
            # for output-target scatter plots
        self.FigSizeScatters = (12,16)
            # for 1-D plots
        self.FigSizeXYPlots = (4,8)
        
        #Snapshot index for drawing Boxplot of prediction deviation
        #starting from 0
        self.SnapIndexForBoxPlot = 0
        
        
    def XLSX_Input_Params(self, XLSXResults):
        Data = [('FOV INDEX', self.FOVIndex), ('Gappiness Percentage (Without Rotor)', self.GappinessPercentageWithoutRotor), ('Gap Size Range', str(self.GapSize))
            , ('Filter Length Scale (X, Y)', str(self.FiltLenX)), ('Filger Type', self.FiltType), ('GPOD nMode', self.nMode), ('GPOD Max Iteration', self.MaxIter)
            , ('PositionFeatureStatus', self.PositionStatus), ('FilteredVelStatus', self.FilteredVelStatus)
            , ('FilteredVortStatus', self.FilteredVortStatus), ('FilteredStrainStatus', self.FilteredStrainStatus), ('OptimizationStatus', self.OptimizationStatus)
            , ('OptCostMetric', self.SVMOptCostMetric), ('ScalerMethod', self.ScalerMethod), ]
        InputDataSheet = XLSXResults.add_worksheet("InputData")
        nRecords = len(Data)
        for Row in range(0, nRecords):
            InputDataSheet.write(Row, 0, Data[Row][0])
            InputDataSheet.write(Row, 1, Data[Row][1])