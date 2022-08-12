# Comparing 2 vector maps
def Config1(VecMap1, VecMap2, VelMapProp, FigureSize, N_Rows, N_Cols):
    from plots import GeneratePlots
    import matplotlib.pyplot as plt
    Fig, ax = plt.subplots(N_Rows,N_Cols)
    for i in range(0,N_Rows):
        for j in range(0,N_Cols):
            
            if N_Rows==1:
                Axes = ax[j]
            elif N_Cols==1:
                Axes = ax[i]
            else:
                Axes = ax[i,j]
            
            if i==0:
                Figure = GeneratePlots(Fig, Axes)
                Figure.ContourPlot(VecMap1, VelMapProp, AxesTitle = 'The Original Velocity Field for u{}'.format(i+1), FigSize=FigureSize)
            elif i==1:
                Figure = GeneratePlots(Fig, Axes)
                Figure.ContourPlot(VecMap2, VelMapProp, AxesTitle = 'The Predicted Velocity Field for u{}'.format(i+1), FigSize=FigureSize)

# Comparing 5 vector maps
def Config2(UID, SnapIndex, Component, VecMap1, VecMap2, VecMap3, VecMap4, VecMap5, VelMapProp, FigureSize, N_Rows, N_Cols, ContourLevels):
    from plots import GeneratePlots
    import matplotlib.pyplot as plt
    Fig, ax = plt.subplots(N_Rows,N_Cols)
    #Fig.suptitle("SnapIndex:{}  Velocity component:{}".format(SnapIndex, Component))
    
    
    Axes = ax[0]
    Figure = GeneratePlots(Fig, Axes)
    Figure.ContourPlot(UID, VecMap1, VelMapProp, AxesTitle = r'Original Velocity: $U_{}^{}$'.format(Component+1, SnapIndex+1), FigSize=FigureSize, ContourLevels = ContourLevels, CMAP = UID.CMAP)
            
    Axes = ax[1]
    Figure = GeneratePlots(Fig, Axes)
    Figure.ContourPlot(UID, VecMap2, VelMapProp, AxesTitle = r'Original Velocity with artificial gaps: $U_{}^{}$'.format(Component+1, SnapIndex+1), FigSize=FigureSize, ContourLevels = ContourLevels, CMAP = UID.CMAP)
    counter = 0
    if UID.MLPlotStatus['SVM'] == True:
        counter+=1
        Axes = ax[1+counter]
        Figure = GeneratePlots(Fig, Axes)
        Figure.ContourPlot(UID, VecMap3, VelMapProp, AxesTitle = r'Predicted Velocity by SVR: $U_{}^{}$'.format(Component+1, SnapIndex+1), FigSize=FigureSize, ContourLevels = ContourLevels, CMAP = UID.CMAP)
    if UID.MLPlotStatus['MLP'] == True:
        counter+=1
        Axes = ax[1+counter]
        Figure = GeneratePlots(Fig, Axes)
        Figure.ContourPlot(UID, VecMap4, VelMapProp, AxesTitle = r'Predicted Velocity by MLP: $U_{}^{}$'.format(Component+1, SnapIndex+1), FigSize=FigureSize, ContourLevels = ContourLevels, CMAP = UID.CMAP)
    if UID.MLPlotStatus['GPOD'] == True:
        counter+=1
        Axes = ax[1+counter]
        Figure = GeneratePlots(Fig, Axes)
        Figure.ContourPlot(UID, VecMap5, VelMapProp, AxesTitle = r'Predicted Velocity by GPOD: $U_{}^{}$'.format(Component+1, SnapIndex+1), FigSize=FigureSize, ContourLevels = ContourLevels, CMAP = UID.CMAP)
    #Axes = ax[0,2].axis('off')
    #plt.tight_layout()
    
def Config3(UID, SnapIndex, Component, StatResults, VecMap1, VecMap2, VecMap3, VecMap4, VecMap5, QuantityLabels, VelMapProp, FigureSize, N_Rows, N_Cols):
    from plots import GeneratePlots
    import matplotlib.pyplot as plt
    Fig, ax = plt.subplots(N_Rows,N_Cols)
    #Fig.suptitle("SnapIndex:{}  Velocity component:{}".format(SnapIndex, Component))
    plt.style.use('seaborn-bright')
    counter=0
    if UID.MLPlotStatus['SVM'] == True:
        counter+=1
        Axes = ax[-1+counter]
        Figure = GeneratePlots(Fig, Axes)
        QLabels = [QuantityLabels[0],QuantityLabels[1],QuantityLabels[2]]
        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE['SVR'], StatResults.PredRMSNAE['SVR'], StatResults.PredMedNAE['SVR'], StatResults.PredR2_Score['SVR'])
        Figure.XYPlotStyle01(VecMap1, VecMap2, VecMap3, QLabels, XLabel = 'PIV x-spacing index', YLabel = r'$U_{}^{}$ (m/s)'.format(Component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize)
    if UID.MLPlotStatus['MLP'] == True:
        counter+=1
        Axes = ax[-1+counter]
        Figure = GeneratePlots(Fig, Axes)
        QLabels = [QuantityLabels[0],QuantityLabels[1],QuantityLabels[3]]
        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE['MLP'], StatResults.PredRMSNAE['MLP'], StatResults.PredMedNAE['MLP'], StatResults.PredR2_Score['MLP'])
        Figure.XYPlotStyle01(VecMap1, VecMap2, VecMap4, QLabels, XLabel = 'PIV x-spacing index', YLabel = r'$U_{}^{}$ (m/s)'.format(Component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize)
    if UID.MLPlotStatus['GPOD'] == True:
        counter+=1
        Axes = ax[-1+counter]
        Figure = GeneratePlots(Fig, Axes)
        QLabels = [QuantityLabels[0],QuantityLabels[1],QuantityLabels[4]]
        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE['GPOD'], StatResults.PredRMSNAE['GPOD'], StatResults.PredMedNAE['GPOD'], StatResults.PredR2_Score['GPOD'])
        Figure.XYPlotStyle01(VecMap1, VecMap2, VecMap5, QLabels, XLabel = 'PIV x-spacing index', YLabel = r'$U_{}^{}$ (m/s)'.format(Component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize)
    #Axes = ax[1,1].axis('off')
    #for i in range(0,10):
    #    plt.tight_layout()
    #FileName = '1D Evaluation_FOV{}_Snap{}_Comp{}.jpg'.format(VelMapProp.FOVIndex+1, SnapIndex+1, Component+1)
    #plt.savefig(fname = FileName, dpi = 200, quality = 95, format='jpg')

def config4(UID, VecMap0, VecMap1, VecMap2, VecMap3, VecMap4, VelMapProp, N_Rows, N_Cols, FigureSize):
    from plots import GeneratePlots
    import matplotlib.pyplot as plt
    Fig, ax = plt.subplots(N_Rows,N_Cols)
    counter=0
    if UID.MLPlotStatus['SVM'] == True:
        counter+=1
        Axes = ax[-1+counter]
        Figure = GeneratePlots(Fig, Axes)
        Figure.QuiverPlot(VecMap0, VecMap1, VecMap2, VelMapProp, FigSize=FigureSize, Model='SVR')
    
    if UID.MLPlotStatus['MLP'] == True:
        counter+=1
        Axes = ax[-1+counter]
        Figure = GeneratePlots(Fig, Axes)
        Figure.QuiverPlot(VecMap0, VecMap1, VecMap3, VelMapProp, FigSize=FigureSize, Model='MLP')
    
    if UID.MLPlotStatus['GPOD'] == True:
        counter+=1
        Axes = ax[-1+counter]
        Figure = GeneratePlots(Fig, Axes)
        Figure.QuiverPlot(VecMap0, VecMap1, VecMap4, VelMapProp, FigSize=FigureSize, Model='GPOD')


def OutputTargetComparisonByScatter(UID, SelectedRegressors, StatResults, SnapIndex, component, N_Rows, N_Cols, FigureSize):
    from plots import GeneratePlots
    import matplotlib.pyplot as plt
    Fig, ax = plt.subplots(N_Rows,N_Cols)
    #Fig.suptitle("SnapIndex:{}  Velocity component:{}".format(SnapIndex, component))
    
#    counter = 0
#    for i in range(0,N_Rows):
#        for j in range(0,N_Cols):
#            
#            if N_Rows==1:
#                Axes = ax[j]
#            elif N_Cols==1:
#                Axes = ax[i]
#            else:
#                Axes = ax[i,j]
                
    counter = -1
    
    for rgrs in SelectedRegressors:
        counter+=1
        RowNo = counter//N_Cols
        ColNo = counter%N_Cols
        Axes = ax[RowNo][ColNo]
        Figure = GeneratePlots(Fig, Axes)
        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE[rgrs], StatResults.PredRMSNAE[rgrs], StatResults.PredMedNAE[rgrs], StatResults.PredR2_Score[rgrs])
        Figure.OutputTargetScatterPlot(StatResults.Output[rgrs], StatResults.Target, XLabel = r'{} Prediction: $U_{}^{}$ (m/s)'.format(rgrs, component+1, SnapIndex+1), YLabel = r'True Value: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize, QuantityLabel=rgrs)
        Axes.set_aspect('equal', 'box')
    
#    if UID.MLPlotStatus['SVM'] == True:
#        counter+=1
#        Axes = ax[-1+counter]
#        Figure = GeneratePlots(Fig, Axes)
#        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE['SVR'], StatResults.PredRMSNAE['SVR'], StatResults.PredMedNAE['SVR'], StatResults.PredR2_Score['SVR'])
#        Figure.OutputTargetScatterPlot(StatResults.Output['SVR'], StatResults.Target, XLabel = r'SVM Prediction: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), YLabel = r'True Value: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize)
#        Axes.set_aspect('equal', 'box')
#    if UID.MLPlotStatus['MLP'] == True:
#        counter+=1
#        Axes = ax[-1+counter]
#        Figure = GeneratePlots(Fig, Axes)
#        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE['MLP'], StatResults.PredRMSNAE['MLP'], StatResults.PredMedNAE['MLP'], StatResults.PredR2_Score['MLP'])
#        Figure.OutputTargetScatterPlot(StatResults.Output['MLP'], StatResults.Target, XLabel = r'MLP Prediction: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), YLabel = r'True Value: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize)
#        Axes.set_aspect('equal', 'box')
#    if UID.MLPlotStatus['GPOD'] == True:
#        counter+=1
#        Axes = ax[-1+counter]
#        Figure = GeneratePlots(Fig, Axes)
#        AxTitle = "MNAE:{0:6.3f}   RMSNAE:{1:6.3f}\nMedNAE:{2:6.3f}   R2 Score:{3:6.3f}".format(StatResults.PredMNAE['GPOD'], StatResults.PredRMSNAE['GPOD'], StatResults.PredMedNAE['GPOD'], StatResults.PredR2_Score['GPOD'])
#        Figure.OutputTargetScatterPlot(StatResults.Output['GPOD'], StatResults.Target, XLabel = r'GPOD Prediction: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), YLabel = r'True Value: $U_{}^{}$ (m/s)'.format(component+1, SnapIndex+1), AxesTitle = AxTitle, FigSize=FigureSize)
#        Axes.set_aspect('equal', 'box')
        
        
        
        
        #Axes = ax[1,1].axis('off')
    #plt.tight_layout()
            
#def VecMapPlot()