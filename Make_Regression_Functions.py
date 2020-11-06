import numpy as np
import pandas as pd

######################################################################

def MakePrediction(Model,\
                   Train_X_Scaled, Val_X_Scaled, Test_X_Scaled,\
                   Scaler_DependVar, MainDF, SplitDataInd):
    
    #  Train_X_Scaled, Val_X_Scaled, Test_X_Scaled should be shaped when RNN is used  
    
    # Input to Model should be numpy array ??
    # If DF convert to numpy array
    if isinstance(Train_X_Scaled, (pd.DataFrame)):
        Train_X_Scaled__IN = Train_X_Scaled.values    
        Val_X_Scaled__IN   = Val_X_Scaled.values         
        Test_X_Scaled__IN  = Test_X_Scaled.values         
    elif isinstance(Train_X_Scaled, (np.ndarray)):    
        Train_X_Scaled__IN = Train_X_Scaled.copy()
        Val_X_Scaled__IN   = Val_X_Scaled.copy()    
        Test_X_Scaled__IN  = Test_X_Scaled.copy()
    
        
    # Take fitted data and make a prediction
    Train_yhat_sld__IN = Model.predict(Train_X_Scaled__IN)
    Val_yhat_sld__IN   = Model.predict(Val_X_Scaled__IN)
    Test_yhat_sld__IN  = Model.predict(Test_X_Scaled__IN)        
    
    # INVERT SCALING
    Train_yhat__IN = Scaler_DependVar.inverse_transform(Train_yhat_sld__IN)
    Val_yhat__IN   = Scaler_DependVar.inverse_transform(Val_yhat_sld__IN)
    Test_yhat__IN  = Scaler_DependVar.inverse_transform(Test_yhat_sld__IN)
    
    
    ### MERGE Fitted and Predicted Data to Main DataFrame
    
    # Take Index of Train, Val and Test sets   
    Index__IN = MainDF.index.to_frame()
    Train_Index__IN, Test_Index__IN = TrainTestSets(Index__IN, SplitDataInd[1])
    Train_Index__IN, Val_Index__IN  = TrainTestSets(Train_Index__IN, SplitDataInd[0])

    # Make DataFrames from yhat:
    Train_yhat_DF__IN = pd.DataFrame(Train_yhat__IN,\
                                     index = Train_Index__IN.index,\
                                     columns = ['Fitted-Train'])
    Val_yhat_DF__IN   = pd.DataFrame(Val_yhat__IN,\
                                     index = Val_Index__IN.index,\
                                     columns = ['Fitted-Validation'])
    Test_yhat_DF__IN  = pd.DataFrame(Test_yhat__IN,\
                                     index = Test_Index__IN.index,\
                                     columns = ['Predicted-Test'])
    
    # Merge MainDF with yhats
    MainDF_WithModelData__IN = MainDF.copy()\
                      .merge(Train_yhat_DF__IN, how='left', on='Date')\
                      .merge(Val_yhat_DF__IN,   how='left', on='Date')\
                      .merge(Test_yhat_DF__IN,  how='left', on='Date')
                      
    return MainDF_WithModelData__IN
