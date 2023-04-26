import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import re

######################################################################

def MakeTSforecast(Data_X, Model, DependentVar,
                   Intecept = False,
                   LagsList = None,
                   Scaler_y = None, Scaler_X = None,
                   Test_or_Forecast = 'Test'):
    
    DF_X = Data_X.copy()
    if Intecept:
        DF_X = sm.add_constant(DF_X)
       #DF_X.insert(0, 'const',  1.0)
        
    # Select Laged Variable which are based on Dependent Variable
    y_LagsList = []
    
    if LagsList is not None:
        y_LagsList = { VarLag:LagNr  for VarLag, LagNr in LagsList.items()\
                          if re.sub(r'_Lag.+', '', VarLag) == DependentVar }
    
    # Make iterative prediction when Dependent Variable has lags:
    if len(y_LagsList) > 0:
        
        yhat = []
        
        for step in range(len(DF_X)):
    
            DF_X_CurrentStep = DF_X.iloc[[step],:]
            yhat_CurrentStep = Model.predict(DF_X_CurrentStep).item()
            yhat.append(yhat_CurrentStep)

            for VarName, LagNr in y_LagsList.items():                
                if step + LagNr < len(DF_X):
                    DF_X.iloc[ step + LagNr,\
                               DF_X.columns.to_list().index(VarName) ] =\
                                yhat_CurrentStep                       
    else:        
        yhat = Model.predict(DF_X)


    # Make Revers Scaling on Obtained Results:
    if Scaler_y is not None:      
        #yhat  = Scaler_y.inverse_transform(yhat)
        yhat  = Scaler_y.inverse_transform( np.array(yhat).reshape(1, -1) ).reshape(-1, 1)
        DF_X  = Scaler_X.inverse_transform(DF_X)
        
        DF_X = pd.DataFrame(DF_X,\
                            index = Data_X.index,\
                            columns = Data_X.columns)
    
        
    # Make DF from yhat:
    yhat_DF  = pd.DataFrame(yhat,\
                            index = Data_X.index,\
                            columns = [f'Predicted-{Test_or_Forecast}'])
    
    
    return (yhat_DF, DF_X)
  

############################################################################################
 
def MakeTSforecastLSTM_XwithDependentVar(Data_X, 
                                         Model, 
                                         DependentVar,
                                         WindowSize = 1,
                                         Scaler_y = None,
                                         Test_or_Forecast = 'Test'):
    '''
    Function iterativly does a prediction on set with dependent variable
    in predictors set.
    
    
    Args:
        Data_X (pd.DataFrame): DataFrame with independent variables
        Model (tensorflow compiled model after fitting): DataFrame with dependent variables
        DependentVar (str): Name of dependent variable
        WindowSize (int): Size of winow used to prepare data
        Scaler_y (sklearn.preprocessing): fitted scaler from sklearn
        Test_or_Forecast (str): String used to name column: Test of Forecast

    Returns:
        yhat_DF (pd.DataFrame): DataFrames with rescaled predicted results 
    '''
    
    DF_X = Data_X.copy()
    NumberOfSteps = int(DF_X.shape[0]/WindowSize)
    
    for column in DF_X.columns:
      if DependentVar in column:
          Dependent_Var = column
    
    yhat = []
    for step in range(NumberOfSteps):
       
       DF_X_CurrentStep = DF_X.copy().iloc[step*WindowSize:(step+1)*WindowSize,:]
       if step>0 and step < NumberOfSteps-1:
           DF_X_CurrentStep[Dependent_Var] = y_for__DF_X_NextStep
       
       X_CurrentStep_shaped = DF_X_CurrentStep.values.reshape(-1,WindowSize,DF_X_CurrentStep.shape[1])
       yhat_CurrentStep = Model.predict(X_CurrentStep_shaped)[0][0]
       yhat.append( yhat_CurrentStep )

       # Prepare y-window for next step
       y_from__DF_X_CurrentStep = DF_X_CurrentStep[Dependent_Var].values
       y_for__DF_X_NextStep = (list(y_from__DF_X_CurrentStep) + [yhat_CurrentStep])[1:]      
       
    # rescale if needed
    if Scaler_y:
        yhat = Scaler_y.inverse_transform(np.array(yhat).reshape(-1, 1))

    # Make DF from yhat:
    Index = DF_X.iloc[WindowSize-1::WindowSize, :].index.to_frame()
    yhat_DF  = pd.DataFrame(yhat,\
                           index = Index.index,\
                           columns = [f'Predicted-{Test_or_Forecast}'])

    return yhat_DF
  
  
############################################################################################
  
def MakeANNfinalData(Model,\
                     Train_X_Scaled, Val_X_Scaled,\
                     Scaler_y,\
                     MainDF,\
                     yhat_Test_DF = None,\
                     yhat_Forecast_DF = None):
    
       #  Train_X_Scaled, Val_X_Scaled, Test_X_Scaled should be shaped when RNN is used  
    
    MainDF_WithModeledData = MainDF.copy()    
        
    # Take fitted data and make a prediction
    yhat_Train_sld = Model.predict(Train_X_Scaled)
    if Val_X_Scaled is not None:
        yhat_Val_sld   = Model.predict(Val_X_Scaled)
    
    # INVERT SCALING
    yhat_Train = Scaler_y.inverse_transform(yhat_Train_sld.reshape(1, -1) ).reshape(-1, 1)
    if Val_X_Scaled is not None:
        yhat_Val   = Scaler_y.inverse_transform(yhat_Val_sld.reshape(1, -1) ).reshape(-1, 1)
  
    
    ### MERGE Fitted and Predicted Data to Main DataFrame
    
    # Make DataFrames from yhat:
    yhat_Train_DF = pd.DataFrame(yhat_Train,\
                                     index = Train_X_Scaled.index,\
                                     columns = ['Fitted-Train'])
    if Val_X_Scaled is not None:
        yhat_Val_DF   = pd.DataFrame(yhat_Val,\
                                     index = Val_X_Scaled.index,\
                                     columns = ['Fitted-Validation'])
        
    
    # Merge MainDF with yhats
    MainDF_WithModeledData = MainDF_WithModeledData\
                      .join(yhat_Train_DF, how='left')
   #                   .merge(yhat_Train_DF, how='left', on='Date')

    if Val_X_Scaled is not None:
        MainDF_WithModeledData = MainDF_WithModeledData\
                      .join(yhat_Val_DF,   how='left')
   #                   .merge(yhat_Val_DF,   how='left', on='Date')

    if yhat_Test_DF is not None:
        
        yhat_Test_DF__IN = yhat_Test_DF.copy()        
        yhat_Test_DF__IN.columns.values[0] = 'Predicted-Test'        
        MainDF_WithModeledData = MainDF_WithModeledData\
                      .join(yhat_Test_DF__IN,  how='left')
   #                 .merge(yhat_Test_DF__IN,  how='left', on='Date')

    if yhat_Forecast_DF is not None:
        
        yhat_Forecast_DF__IN = yhat_Forecast_DF.copy()
        yhat_Forecast_DF__IN.columns.values[0] = 'Forecast'
        MainDF_WithModeledData = pd.concat( [MainDF_WithModeledData,\
                                               yhat_Forecast_DF__IN] )
                      
                      
    return MainDF_WithModeledData


############################################################################
############################################################################
############################################################################


def MakeFinalDataFull(Model,\
                      Train_X_Scaled, Val_X_Scaled,\
                      Scaler_y,\
                      MainDF,\
                      TestSplitInd, ValSplitInd,\
                      yhat_Test_DF = None,\
                      yhat_Forecast_DF = None):
    
    #  Train_X_Scaled, Val_X_Scaled, Test_X_Scaled should be shaped when RNN is used  
    
    MainDF_WithModeledData = MainDF.copy()
        
    # Take fitted data and make a prediction
    yhat_Train_sld = Model.predict(Train_X_Scaled)
    yhat_Val_sld   = Model.predict(Val_X_Scaled)
    
    # INVERT SCALING
    yhat_Train = Scaler_y.inverse_transform(yhat_Train_sld)
    yhat_Val   = Scaler_y.inverse_transform(yhat_Val_sld)
  
    
    ### MERGE Fitted and Predicted Data to Main DataFrame
    
    # Take Index of Train, Val and Test sets 
    Index__MainDF = MainDF_WithModeledData.index.to_frame()
    Index_Train, Index_Test = TrainTestSets(Index__MainDF, TestSplitInd)
    Index_Train, Index_Val  = TrainTestSets(Index_Train, ValSplitInd)

    # Make DataFrames from yhat:
    yhat_Train_DF = pd.DataFrame(yhat_Train,\
                                     index = Index_Train.index,\
                                     columns = ['Fitted-Train'])
    yhat_Val_DF   = pd.DataFrame(yhat_Val,\
                                     index = Index_Val.index,\
                                     columns = ['Fitted-Validation'])
        
    
    # Merge MainDF with yhats
    MainDF_WithModeledData = MainDF_WithModeledData\
                               .join(yhat_Train_DF, how='left')\
                               .join(yhat_Val_DF,   how='left')
    #                          .merge(yhat_Train_DF, how='left', on='Date')\
    #                          .merge(yhat_Val_DF,   how='left', on='Date')
                                            
    if yhat_Test_DF is not None:
        
        yhat_Test_DF__IN = yhat_Test_DF.copy()
        yhat_Test_DF__IN.columns.values[0] = 'Predicted-Test'
        MainDF_WithModeledData = MainDF_WithModeledData\
                       .join(yhat_Test_DF__IN,  how='left')
        #              .merge(yhat_Test_DF__IN,  how='left', on='Date')
        
    if yhat_Forecast_DF is not None:
        
        yhat_Forecast_DF__IN = yhat_Forecast_DF.copy()
        yhat_Forecast_DF__IN.columns.values[0] = 'Forecast'
        MainDF_WithModeledData = pd.concat( [MainDF_WithModeledData,\
                                               yhat_Forecast_DF__IN] )
                      
                      
    return MainDF_WithModeledData


###########################################################

def MakeRegressStatModelsWithFormula(TrainSet, TestSet, formula):
    
    
    Train = TrainSet.copy()
    Test = TestSet.copy()
    
    Model = smf.ols(formula = formula, data = TrainSet)
    DependedVariable = Model.endog_names
    
    ModelFitted = Model.fit()
    
    Train['Fitted'] = ModelFitted.fittedvalues
    Test['Predicted-Test'] = ModelFitted.predict(TestSet)
    
    plot = Train.append(Test)\
                [[DependedVariable, 'Fitted', 'Predicted-Test']].plot()
                
    print(plot)
    
    CalculateR2andR2adjForSatModelsWithFormula(\
                                Test, DependedVariable, ModelFitted)

    print('MAE: '+str( MAE(Test[DependedVariable],\
                           Test['Predicted-Test'] )) )
    print('RSME: '+str( RSME(Test[DependedVariable],\
                             Test['Predicted-Test'] )) )   
        
    
    FullData = Train.append(Test)
    
    return (FullData, ModelFitted)
