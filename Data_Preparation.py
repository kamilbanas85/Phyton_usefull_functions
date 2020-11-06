import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


############################################################################################

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


############################################################################################

def ScaleThenConvertArrayToDF(BasicDF, Scaler):
    
    BasicArray_sld = Scaler.transform(BasicDF)
    
    # convert to DataFrames becouse After scaling data stracture is array
    BasicDF_sld = pd.DataFrame(BasicArray_sld,\
                               index   = BasicDF.index,\
                               columns = BasicDF.columns)
            
    return BasicDF_sld


###############################################

def PrepareDataForRegression(DataDF, DependentVar, IndependentVar,\
                             SplitDataIndicator, 
                             DummyForCol = None,
                             ScalerType = None,
                             ScalerRange = (0,1)):
    # SplitDataIndicator <- list
    # DependentVar <- str, IndependentVar <- list of str
    # ScalerType <- 'MinMax' or 'Standard'
    # ScalerRange = (0,1), or (-1,1)    - tuple
    
    TestSplitInd = None
    ValSplitInd = None
    DataToReturn = []
    
    # Set slit date for valid and test sets:
    if len(SplitDataIndicator) == 2:
        TestSplitInd = SplitDataIndicator[1]
        ValSplitInd  = SplitDataIndicator[0]
    elif len(SplitDataIndicator) == 1:
        TestSplitInd  = SplitDataIndicator[0]
    
    # Select columns of DataFrame:    
    DF = DataDF.copy().loc[:, [DependentVar] + IndependentVar]
    
    # CREATE DUMMY VARIABLES FOR 'DummyForCol' COLUMNS
    if DummyForCol is not None:
        DF = pd.get_dummies(DF, columns = [DummyForCol],\
                                prefix = [DummyForCol+'_'], drop_first=True )
        
                
    ### SPLIT INTO DEPENDENT AND INDEPANEDNT VARIABLES
    DF_y = DF.loc[:, [DependentVar] ]
    DF_X = DF.drop(columns= [DependentVar] )
     
          
    ### DIVIDE INTO TRAIN, VALIDATION AND TEST SETS    
    TrainDF_y, TestDF_y = TrainTestSets(DF_y, TestSplitInd)
    TrainDF_X, TestDF_X = TrainTestSets(DF_X, TestSplitInd)
    if ValSplitInd is not None:
        TrainDF_y, ValDF_y  = TrainTestSets(TrainDF_y, ValSplitInd)
        TrainDF_X, ValDF_X  = TrainTestSets(TrainDF_X, ValSplitInd)


    # Return data if NO scaling or SCALE DATA:
    if ScalerType is None:
        # Create list of DF to return
        DataToReturn = [TrainDF_X, TrainDF_y,\
                        TestDF_X, TestDF_y]
        if ValSplitInd is not None:
            DataToReturn =\
                DataToReturn[:2] + [ValDF_X, ValDF_y] + DataToReturn[2:]
                
    elif ScalerType is not None:
        # Define Scalers:        
        if ScalerType == 'MinMax':    
            scaler_y = MinMaxScaler( feature_range = ScalerRange )
            scaler_X = MinMaxScaler( feature_range = ScalerRange )        
        elif ScalerType == 'Standard':    
            scaler_y = StandardScaler()
            scaler_X = StandardScaler()
    
        # Fit scalers on Trains Sets:
        scaler_y = scaler_y.fit(TrainDF_y)
        scaler_X = scaler_X.fit(TrainDF_X)    

        # Scale And Convert to DataFrame Train and Test sets:        
        Train_y_sld = ScaleThenConvertArrayToDF(TrainDF_y, scaler_y)
        Train_X_sld = ScaleThenConvertArrayToDF(TrainDF_X, scaler_X)

        Test_y_sld = ScaleThenConvertArrayToDF(TestDF_y, scaler_y)
        Test_X_sld = ScaleThenConvertArrayToDF(TestDF_X, scaler_X)
        
        # Create data to return:
        DataToReturn = [Train_X_sld, Train_y_sld,\
                        Test_X_sld, Test_y_sld,\
                        scaler_X, scaler_y]
        
        if ValSplitInd is not None:
            # Scale And Convert to DataFrame Val sets        
            Val_y_sld = ScaleThenConvertArrayToDF(ValDF_y, scaler_y)
            Val_X_sld = ScaleThenConvertArrayToDF(ValDF_X, scaler_X)
            
            # Extend data to return:
            DataToReturn =\
                DataToReturn[:2] + [Val_X_sld, Val_y_sld] + DataToReturn[2:]
          
                
    return tuple(DataToReturn)
        
        
