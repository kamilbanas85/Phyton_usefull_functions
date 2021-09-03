import numpy as np
import pandas as pd


import datetime
import matplotlib.pyplot as plt


################################################################


def CVtimeSeries(DF_X, DF_y, Model,
                 DependentVar,
                 Intecept = False,
                 LagsList = None,
                 Scaler_y = None, Scaler_X = None,
                 PlotResults = True):
    

    X_set = DF_X.copy()
    y_set = DF_y.copy()
   
    GetCVdatesresults = GetCVdates(X_set)
    End_train_dates  = GetCVdatesresults['train_dates']
    delta = GetCVdatesresults['delta']

    MAEs = []
    MAPEs = []
     
    if Intecept:
        X_set = sm.add_constant(X_set)

    if (PlotResults):       
        fig, axs = plt.subplots( len(End_train_dates) , sharex = True, sharey = True)
 


    for i, train_time in  enumerate(End_train_dates):
               
        test_time = train_time + datetime.timedelta(days=1)

        X_train__Current = X_set.copy().loc[:train_time]
        y_train__Current = y_set.copy().loc[:train_time]

        X_test__Current  = X_set.copy().loc[test_time:test_time+delta]
        y_test__Current  = y_set.copy().loc[test_time:test_time+delta]

        ModelFitted =  Model(y_train__Current, X_train__Current).fit()

    
        yhat_Test__Current, X_Test_withLags__Current = \
                MakeTSforecast(X_test__Current, ModelFitted,\
                               DependentVar = DependentVar,
                               Intecept = Intecept,
                               LagsList = LagsList,
                               Scaler_y = None, Scaler_X = None,
                               Test_or_Forecast = 'Test')        
               
        MAEs.append( MAE(y_test__Current, yhat_Test__Current) )
        MAPEs.append( MAPE(y_test__Current, yhat_Test__Current))        
        
        if (PlotResults):
            
            TrainPlot,    = axs[i].plot(y_train__Current,   color = 'black')
            TestPlot,     = axs[i].plot(y_test__Current,    color = 'blue')
            ForecastPlot,  = axs[i].plot(yhat_Test__Current, color = 'red')
            
            textAdnotation = (f"""MAE: {MAE(y_test__Current, yhat_Test__Current)}\n"""
                              f"""MAPE: {MAPE(y_test__Current, yhat_Test__Current)}""")
            
            props = dict(boxstyle='round', facecolor='skyblue', alpha=0.5)
            axs[i].text(0.01, 0.88, textAdnotation, transform=axs[i].transAxes,\
                        fontsize=16, verticalalignment='top', bbox=props)
  
    if (PlotResults):

        plt.rcParams["font.size"] = "18"
    
        fig.text(0.5, 0.04, 'Data', fontsize = 20, ha='center')
        fig.text(0.08, 0.5, 'Price [ PLN/MWh ]', fontsize = 20, va='center', rotation='vertical')

        fig.legend((TrainPlot,TestPlot, ForecastPlot),     # The line objects
                   ('Messure - Train', 'Messure - Test','Forcast'),
                   fontsize = 16, 
                   bbox_to_anchor=[0.5,0.93], 
                   loc='center',                   
                   ncol=3,   # Position of legend
                   borderaxespad=0.0)  
        plt.show()

    
    return {'MAE':MAEs, 'MAPE':MAPEs} 
