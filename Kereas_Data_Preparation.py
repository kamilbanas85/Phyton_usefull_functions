import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler



##############################################

def TrainTestSets(DFdata, SplitIndicator):
    
    DFdataTrain = pd.DataFrame()
    DFdataTest = pd.DataFrame()
    
    if isinstance(SplitIndicator, (str)):
        DFdataTrain, DFdataTest =\
                      DFdata[DFdata.index <  SplitIndicator],\
                      DFdata[DFdata.index >= SplitIndicator]            
    elif isinstance(SplitIndicator, (float)):
        DFdataTrain, DFdataTest =\
            np.split(DFdata, [int(SplitIndicator* len(DFdata))] )
            
    return (DFdataTrain, DFdataTest)



####################################################################################

def PrepareDataForANN(DataDF, DependentVar, IndependentVar, DummyForCol,
                      SplitDataInd):
    
    DataDFin = DataDF.copy().loc[:, [DependentVar] + IndependentVar]
    
    
    DataDFin = pd.get_dummies(DataDFin, columns=[DummyForCol],\
                                prefix=[DummyForCol+'_'], drop_first=True )
        
        
    ### SPLIT INTO INPUT DEPENDENT AND INDEPANEDNT VARIABLES
    DataDFin_y = DataDFin.loc[:, [DependentVar] ]
    DataDFin_X = DataDFin.drop(columns= [DependentVar] )
     
          
    ### DIVIDE INTO TRAIN, VALIDATION AND TEST SETS
    TrainDFin_y, TestDFin_y = TrainTestSets(DataDFin_y, SplitDataInd[1])
    TrainDFin_y, ValDFin_y  = TrainTestSets(TrainDFin_y, SplitDataInd[0])

    TrainDFin_X, TestDFin_X = TrainTestSets(DataDFin_X, SplitDataInd[1])
    TrainDFin_X, ValDFin_X  = TrainTestSets(TrainDFin_X, SplitDataInd[0])


    ### SCALE DATA
    # Fit scaler on Train Set
    
    scalerIn_IndepVars = MinMaxScaler(feature_range=(0, 1))
    scalerIn_DepVar    = MinMaxScaler(feature_range=(0, 1))

    scalerIn_IndepVars = scalerIn_IndepVars.fit(TrainDFin_X)    
    scalerIn_DepVar    = scalerIn_DepVar.fit(TrainDFin_y)

    # Transform train    
    TrainIN_X_sld = scalerIn_IndepVars.transform(TrainDFin_X)
    TrainIN_y_sld = scalerIn_DepVar.transform(TrainDFin_y)
    # Transform valid    
    ValIN_X_sld = scalerIn_IndepVars.transform(ValDFin_X)
    ValIN_y_sld = scalerIn_DepVar.transform(ValDFin_y)    
    # Transform test    
    TestIN_X_sld = scalerIn_IndepVars.transform(TestDFin_X)
    TestIN_y_sld = scalerIn_DepVar.transform(TestDFin_y)
    
    # After scaling data stracture is array
    # Create DataFrame with colNames                                
    TrainDFin_X_sld = pd.DataFrame(TrainIN_X_sld,\
                                   index   = TrainDFin_X.index,\
                                   columns = TrainDFin_X.columns)
    TrainDFin_y_sld = pd.DataFrame(TrainIN_y_sld,\
                                   index   = TrainDFin_y.index,\
                                   columns = TrainDFin_y.columns)   

    ValDFin_X_sld = pd.DataFrame(ValIN_X_sld,\
                                   index   = ValDFin_X.index,\
                                   columns = ValDFin_X.columns)
    ValDFin_y_sld = pd.DataFrame(ValIN_y_sld,\
                                 index   = ValDFin_y.index,\
                                 columns = ValDFin_y.columns)
        
    TestDFin_X_sld = pd.DataFrame(TestIN_X_sld,\
                                  index   = TestDFin_X.index,\
                                  columns = TestDFin_X.columns)
    TestDFin_y_sld = pd.DataFrame(TestIN_y_sld,\
                                  index   = TestDFin_y.index,\
                                  columns = TestDFin_y.columns) 
        
    return (TrainDFin_X_sld, TrainDFin_y_sld,\
            ValDFin_X_sld, ValDFin_y_sld,\
            TestDFin_X_sld, TestDFin_y_sld,\
            scalerIn_DepVar)
        
        
        
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
