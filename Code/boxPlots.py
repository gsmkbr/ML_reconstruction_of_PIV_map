def BoxPlot(Data, SelectedRegressors, FOVS, UID):
    Labels = []
    for FovIndex in FOVS:
        for rgrs in SelectedRegressors:
            Labels.append(rgrs)
            #Labels.append('FOV'+str(FovIndex)+'\n'+rgrs)
#        Labels.append('FOV'+str(FovIndex)+'\nSVR')
#        Labels.append('FOV'+str(FovIndex)+'\nMLP')
#        if UID.MLPlotStatus['GPOD'] == True:
#            Labels.append('FOV'+str(FovIndex)+'\nGPOD')
    
    import matplotlib.pyplot as plt

    for index in range(1,4):
        DataForBoxPlot = Data[str(index)]
        fig, ax = plt.subplots()
        ax.boxplot(DataForBoxPlot, showfliers=False, labels=Labels)
        if UID.PredicitonErrorType == 'Absolute':
            YLabel= r'$\Delta U_{} (m/s)$'.format(index)
        elif UID.PredicitonErrorType == 'Normalized':
            YLabel= r'$\Delta U_{}^*$'.format(index)
        
        ax.set_ylabel(YLabel, fontsize=14)
        #ax.set_xticklabels(Labels, fontsize=13)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=12)
        
        if UID.PredicitonErrorType == 'Absolute':
            ax.set_ylim([-0.1,3.0])
        
        ax.set_xticklabels(Labels, rotation=45, ha='right', rotation_mode='anchor')
        fig.tight_layout()
        plt.show()
        
def BoxPlot2(Data, SelectedRegressors, FOVS, UID):
    Labels = []
    for FovIndex in FOVS:
        for rgrs in SelectedRegressors:
            Labels.append(rgrs)
            #Labels.append('FOV'+str(FovIndex)+'\n'+rgrs)
#    Labels = []
#    for FovIndex in FOVS:
#        Labels.append('FOV'+str(FovIndex)+'\nSVR')
#        Labels.append('FOV'+str(FovIndex)+'\nMLP')
#        if UID.MLPlotStatus['GPOD'] == True:
#            Labels.append('FOV'+str(FovIndex)+'\nGPOD')
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.boxplot(Data, showfliers=False, labels=Labels)
    YLabel= r'$\Delta \mathit{MC}^*$' #r'$\Delta S (1/s)$'
    
    ax.set_ylabel(YLabel, fontsize=14)
    #ax.set_xticklabels(Labels, fontsize=13)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    
    ax.set_xticklabels(Labels, rotation=45, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    plt.show()
    
#def BoxPlot3(Data, FOVS, UID):
#    Labels = []
#    for FovIndex in FOVS:
#        Labels.append('FOV'+str(FovIndex)+'\nSVR')
#        Labels.append('FOV'+str(FovIndex)+'\nMLP')
#        if UID.MLPlotStatus['GPOD'] == True:
#            Labels.append('FOV'+str(FovIndex)+'\nGPOD')
#    
#    import matplotlib.pyplot as plt
#
#    for index in range(1,4):
#        DataForBoxPlot = Data[str(index)]
#        fig, ax = plt.subplots()
#        ax.boxplot(DataForBoxPlot, showfliers=False, labels=Labels)
#        if UID.PredicitonErrorType == 'Absolute':
#            YLabel= r'$\Delta U_{} (m/s)$'.format(index)
#        elif UID.PredicitonErrorType == 'Normalized':
#            YLabel= r'$\Delta U_{}^*$'.format(index)
#        
#        ax.set_ylabel(YLabel, fontsize=14)
#        #ax.set_xticklabels(Labels, fontsize=13)
#        ax.tick_params(axis="x", labelsize=14)
#        ax.tick_params(axis="y", labelsize=12)
#        
#        if UID.PredicitonErrorType == 'Absolute':
#            ax.set_ylim([-0.1,3.0])
#        
#        plt.show()