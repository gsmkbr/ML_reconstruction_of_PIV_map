i = 2
ContMetricMLP2 = {}
ContMetricSVR2 = {}
ContMetricOrig2 = {}

ContAbsDeviationSVR2 = []
ContAbsDeviationMLP2 = []

ContMetricMLP2['FOV'+str(i)]=ContMetricMLP['FOV'+str(i+1)].copy()
ContMetricSVR2['FOV'+str(i)]=ContMetricSVR['FOV'+str(i+1)].copy()
ContMetricOrig2['FOV'+str(i)]=ContMetricOrig['FOV'+str(i+1)].copy()
ContAbsDeviationSVR2['FOV'+str(i)] = ContAbsDeviationSVR['FOV'+str(i+1)].copy()
ContAbsDeviationMLP2['FOV'+str(i)] = ContAbsDeviationMLP['FOV'+str(i+1)].copy()

ContAbsDeviationSVR2.append(ContAbsDeviationSVR['FOV'+str(i)])
ContAbsDeviationMLP2.append(ContAbsDeviationMLP['FOV'+str(i)])



FOVS = [1,2,3,4]
DataForBoxPlot3 = []
from plots import GeneratePlots
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
from boxPlots import BoxPlot2
for FOVIndex in FOVS:
    DataForBoxPlot3.append(ContAbsDeviationSVR3['FOV'+str(FOVIndex)]/(ContMetricOrig3['FOV'+str(FOVIndex)].std()))
    DataForBoxPlot3.append(ContAbsDeviationMLP3['FOV'+str(FOVIndex)]/(ContMetricOrig3['FOV'+str(FOVIndex)].std()))
    Fig, ax = plt.subplots(1,1)
    Figure = GeneratePlots(Fig, ax)
#    AxTitle = {'SVR': 'MAE:{0:7.2f} (1/s)\nMSE:{1:7.2f} (1/s)\nMedAE:{2:7.2f} (1/s)\n'.format(mean_absolute_error(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricSVR['FOV'+str(FOVIndex)]),
#                                                           sqrt(mean_squared_error(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricSVR['FOV'+str(FOVIndex)])),
#                                                           median_absolute_error(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricSVR['FOV'+str(FOVIndex)]))
#                        +r'$R_2-score:$'+'{0:6.3f}'.format(r2_score(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricSVR['FOV'+str(FOVIndex)]))
#                , 'MLP': 'MAE:{0:7.2f} (1/s)\nMSE:{1:7.2f} (1/s)\nMedAE:{2:7.2f} (1/s)\n'.format(mean_absolute_error(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricMLP['FOV'+str(FOVIndex)]),
#                                                           sqrt(mean_squared_error(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricMLP2['FOV'+str(FOVIndex)])),
#                                                           median_absolute_error(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricMLP['FOV'+str(FOVIndex)]))
#                        +r'$R_2-score:$'+'{0:6.3f}'.format(r2_score(ContMetricOrig['FOV'+str(FOVIndex)], ContMetricMLP['FOV'+str(FOVIndex)]))}
    AxTitle = 'FOV'+str(FOVIndex)+'\nMAE: ({0:.2f}, {1:.2f}) 1/s\nRMSE: ({2:.2f}, {3:.2f}) 1/s\nMedAE: ({4:.2f}, {5:.2f}) 1/s\n'.format(mean_absolute_error(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricSVR3['FOV'+str(FOVIndex)]),
                        mean_absolute_error(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricMLP3['FOV'+str(FOVIndex)]),
                        sqrt(mean_squared_error(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricSVR3['FOV'+str(FOVIndex)])),
                        sqrt(mean_squared_error(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricMLP3['FOV'+str(FOVIndex)])),
                        median_absolute_error(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricSVR3['FOV'+str(FOVIndex)]),
                        median_absolute_error(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricMLP3['FOV'+str(FOVIndex)]))+r'$R_2-score:$'+' ({0:.3f}, {1:.3f})'.format(r2_score(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricSVR3['FOV'+str(FOVIndex)]),
                        r2_score(ContMetricOrig3['FOV'+str(FOVIndex)], ContMetricMLP3['FOV'+str(FOVIndex)]))
    FigureSize = (6,6)
    Figure.OutputTargetScatterPlot(ContMetricMLP3['FOV'+str(FOVIndex)], ContMetricOrig3['FOV'+str(FOVIndex)], XLabel = r'Predicted $\mathit{MC}$ (1/s)', YLabel = r'Original $\mathit{MC}$ (1/s)', AxesTitle = AxTitle, FigSize=FigureSize, COLOR = 'g', ALPHA = 0.6, Slope45=False, LabelSize=14, TitleSize=14, TicLabelSize=12, QuantityLabel='MLP')
    Figure.OutputTargetScatterPlot(ContMetricSVR3['FOV'+str(FOVIndex)], ContMetricOrig3['FOV'+str(FOVIndex)], XLabel = r'Predicted $\mathit{MC}$ (1/s)', YLabel = r'Original $\mathit{MC}$ (1/s)', AxesTitle = AxTitle, FigSize=FigureSize, COLOR = 'b', ALPHA = 0.6, Slope45=True, LabelSize=14, TitleSize=14, TicLabelSize=12, QuantityLabel='SVR')
    
BoxPlot2(DataForBoxPlot3, FOVS)