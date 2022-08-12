class GeneratePlots:
    def __init__(self, Fig, Axes):
        self.Fig = Fig
        self.Axes = Axes
        
    def ContourPlot(self, UID, VecMap, VelMapProp, AxesTitle = ' ', FigSize=(6,4), ContourLevels = [0,30], CMAP = "RdBu_r"):
        import matplotlib.pyplot as plt
        import numpy as np
        self.NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        self.NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        self.Fig.set_size_inches(FigSize)
        if UID.AxTitleStatus == True:
            self.Axes.set_title(AxesTitle)
        x=np.arange(1,self.NCol+1)
        y=np.arange(1,self.NRow+1)
        X,Y=np.meshgrid(x,y)
        CNTRF = self.Axes.contourf(X,Y,VecMap, levels = ContourLevels, cmap = CMAP)
        if UID.CMAPStatus == True:
            CLRBar = self.Fig.colorbar(CNTRF, ax = self.Axes)
            CLRBar.set_label('m/s')
        self.Axes.set_aspect('equal', 'box')
        if UID.DrawHLine == True:
            self.Axes.plot([x[0],x[-1]], [UID.HLinePosition,UID.HLinePosition], linestyle = '--', color = 'lime')
        if UID.AxLabelStatus == False:
            self.Axes.set_xticklabels([])
            self.Axes.set_yticklabels([])
            self.Axes.set_xticks([])
            self.Axes.set_yticks([])
        
    def QuiverPlot(self, VecMap0, VecMap1, VecMap2, VelMapProp, FigSize=(6,4), Model='SVR'):
        import matplotlib.pyplot as plt
        import numpy as np
        self.NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
        self.NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
        self.Fig.set_size_inches(FigSize)
        x=np.arange(1,self.NCol+1)
        y=np.arange(1,self.NRow+1)
        X,Y=np.meshgrid(x,y)
        skip=(slice(None,None,4),slice(None,None,4))
        #QV = self.Axes.quiver(X[skip], Y[skip], VecMap1['x'][skip], VecMap1['y'][skip], color='blue')   #, units='inches', scale_units='inches', scale=FigSize[0]/0.02, color='blue', width=0.01, headwidth=7., headlength=7)
        #QV = self.Axes.quiver(X[skip], Y[skip], VecMap2['x'][skip], VecMap2['y'][skip], color='red')
        
        NonZeroIndex = np.where(VecMap0['x']!=0)
        NonZeroBool = np.copy(VecMap0['x'])
        NonZeroBool[NonZeroIndex]=1
        
        from matplotlib import colors
        cmap = colors.ListedColormap(['white', 'yellow'])
        self.Axes.contourf(X,Y,NonZeroBool, cmap=cmap)
        QV = self.Axes.quiver(X[skip], Y[skip], VecMap1['x'][skip], VecMap1['y'][skip], color='blue', alpha=0.8, units='width', scale_units='width', scale=400, width=0.005, headwidth=3, headlength=5)
        self.Axes.quiverkey(QV, X=0.05, Y=1.05, U=20, label='Original Vel. (scale:20 m/s)', labelpos='E')
        QV = self.Axes.quiver(X[skip], Y[skip], VecMap2['x'][skip], VecMap2['y'][skip], color='red', alpha=0.8, units='width', scale_units='width', scale=400, width=0.005, headwidth=3, headlength=5)
        self.Axes.quiverkey(QV, X=0.6, Y=1.05, U=20, label=Model+' Vel. (scale:20 m/s)', labelpos='E')
        
    def OutputTargetScatterPlot(self, Output, Target, XLabel, YLabel, AxesTitle, FigSize, COLOR = 'b', ALPHA = 0.4, Slope45 = True, LabelSize = 12, TitleSize = 12, TicLabelSize = 11, QuantityLabel='', PlotRange = 'Auto', LegendStatus = 'On'):
        import matplotlib.pyplot as plt
        import numpy as np
        if PlotRange == 'Auto':
            MinVal = min(np.min(Output), np.min(Target))-1
            MaxVal = max(np.max(Output), np.max(Target))+1
        else:
            MinVal = -1000
            MaxVal = 1000
        self.Fig.set_size_inches(FigSize)
        self.Axes.scatter(Output, Target, c=COLOR, alpha=ALPHA, label=QuantityLabel)
        if Slope45 == True:
            self.Axes.plot([MinVal,MaxVal],[MinVal,MaxVal], c='r', linewidth=2)
        self.Axes.set_xlabel(XLabel, fontsize=LabelSize)
        self.Axes.set_ylabel(YLabel, fontsize=LabelSize)
        self.Axes.set_title(AxesTitle, loc='left', fontsize=TitleSize)
        self.Axes.tick_params(axis="x", labelsize=TicLabelSize)
        self.Axes.tick_params(axis="y", labelsize=TicLabelSize)
        if LegendStatus == 'On':
            self.Axes.legend(fontsize=14)
        if PlotRange != 'Auto':
            self.Axes.set_xlim(-1000,1000)
            self.Axes.set_ylim(-1000,1000)
        self.Axes.set_aspect('equal')
        #self.Axes.tight_layout()
        
    def XYPlotStyle01(self, VecMap1, VecMap2, VecMap3, QuantityLabels, XLabel, YLabel, AxesTitle, FigSize=(6,4)):
        import matplotlib.pyplot as plt
        import numpy as np
        XData = np.arange(0, VecMap1.size)
        self.Fig.set_size_inches(FigSize)
        Transparency_orig=1.0
        Transparency_gaps=1.0
        Transparency_model=0.8
        self.Axes.scatter(XData, VecMap1, c='b', s=40, alpha = Transparency_orig, label=QuantityLabels[0])
        self.Axes.scatter(XData, VecMap2, c='c', s=20, alpha = Transparency_gaps, label=QuantityLabels[1])
        #self.Axes.scatter(XData, VecMap3, c='r' , s=20, alpha = Transparency_model, label=QuantityLabels[2])
        self.Axes.plot(XData, VecMap3, c='r' , alpha = Transparency_model, label=QuantityLabels[2])
        self.Axes.set_xlabel(XLabel)
        self.Axes.set_ylabel(YLabel)
        self.Axes.set_title(AxesTitle)
        self.Axes.legend()
        
#    def VectorPlot(self, VecMap, VelMapProp, AxesTitle = ' ', FigSize=(6,4)):
#        import matplotlib.pyplot as plt
#        import numpy as np
#        self.NRow = VelMapProp.NRow[str(VelMapProp.FOVIndex)]
#        self.NCol = VelMapProp.NCol[str(VelMapProp.FOVIndex)]
#        self.Fig.set_size_inches(FigSize)
#        self.Axes.set_title(AxesTitle)
#        x=np.arange(1,self.NCol+1)
#        y=np.arange(1,self.NRow+1)
#        X,Y=np.meshgrid(x,y)
#        self.Axes.quiver(X,Y,VecMap)