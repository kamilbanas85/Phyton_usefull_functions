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
    
    DataDF__IN = DataDF.copy().loc[:, [DependentVar] + IndependentVar]
    
    # CREATE DUMMY VARIABLES FOR SELECTED COLUMNS
    DataDF__IN = pd.get_dummies(DataDF__IN, columns=[DummyForCol],\
                                prefix=[DummyForCol+'_'], drop_first=True )
        
        
    ### SPLIT INTO INPUT DEPENDENT AND INDEPANEDNT VARIABLES
    DataDF_y__IN = DataDF__IN.loc[:, [DependentVar] ]
    DataDF_X__IN = DataDF__IN.drop(columns= [DependentVar] )
     
          
    ### DIVIDE INTO TRAIN, VALIDATION AND TEST SETS
    TrainDF_y__IN, TestDF_y__IN = TrainTestSets(DataDF_y__IN, SplitDataInd[1])
    TrainDF_y__IN, ValDF_y__IN  = TrainTestSets(TrainDF_y__IN, SplitDataInd[0])

    TrainDF_X__IN, TestDF_X__IN = TrainTestSets(DataDF_X__IN, SplitDataInd[1])
    TrainDF_X__IN, ValDF_X__IN  = TrainTestSets(TrainDF_X__IN, SplitDataInd[0])


    ### SCALE DATA
    # Fit scaler on Train Set
    
    scaler_IndepVars__IN = MinMaxScaler(feature_range=(0, 1))
    scaler_DepVar__IN    = MinMaxScaler(feature_range=(0, 1))

    scaler_IndepVars__IN = scaler_IndepVars__IN.fit(TrainDF_X__IN)    
    scaler_DepVar__IN    = scaler_DepVar__IN.fit(TrainDF_y__IN)

    # Transform train    
    Train_X_sld__IN = scaler_IndepVars__IN.transform(TrainDF_X__IN)
    Train_y_sld__IN = scaler_DepVar__IN.transform(TrainDF_y__IN)
    # Transform valid    
    Val_X_sld__IN = scaler_IndepVars__IN.transform(ValDF_X__IN)
    Val_y_sld__IN = scaler_DepVar__IN.transform(ValDF_y__IN)    
    # Transform test    
    Test_X_sld__IN = scaler_IndepVars__IN.transform(TestDF_X__IN)
    Test_y_sld__IN = scaler_DepVar__IN.transform(TestDF_y__IN)
    
    # CONVERT TO DataFrames
    # After scaling data stracture is array, so
    # Create DataFrame with colNames                                
    TrainDF_X_sld__IN = pd.DataFrame(Train_X_sld__IN,\
                                     index   = TrainDF_X__IN.index,\
                                     columns = TrainDF_X__IN.columns)
    TrainDF_y_sld__IN = pd.DataFrame(Train_y_sld__IN,\
                                     index   = TrainDF_y__IN.index,\
                                     columns = TrainDF_y__IN.columns)   

    ValDF_X_sld__IN = pd.DataFrame(Val_X_sld__IN,\
                                   index   = ValDF_X__IN.index,\
                                   columns = ValDF_X__IN.columns)
    ValDF_y_sld__IN = pd.DataFrame(Val_y_sld__IN,\
                                   index   = ValDF_y__IN.index,\
                                   columns = ValDF_y__IN.columns)
        
    TestDF_X_sld__IN = pd.DataFrame(Test_X_sld__IN,\
                                    index   = TestDF_X__IN.index,\
                                    columns = TestDF_X__IN.columns)
    TestDF_y_sld__IN = pd.DataFrame(Test_y_sld__IN,\
                                    index   = TestDF_y__IN.index,\
                                    columns = TestDF_y__IN.columns) 
        
    return (TrainDF_X_sld__IN, TrainDF_y_sld__IN,\
            ValDF_X_sld__IN, ValDF_y_sld__IN,\
            TestDF_X_sld__IN, TestDF_y_sld__IN,\
            scaler_IndepVars__IN, scaler_DepVar__IN)
        
        
        
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
