import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import datetime
import re


#########################################################

def DevideOnXandY_CreateDummies(DataDF,
                                DependentVar,
                                IndependentVar,
                                DummyForCol = None):
    '''
    Function devide data on predictors 'X' and taget 'y' sets. 
    If needed it add dummies into predictors. 
  
    
    Args:
        Data_X (pd.DataFrame): DataFrame with all vaiables
        DependentVar (str): Name of dependent var
        IndependentVar (list(str)): List of Independent variables
        DummyForCol (list(str)): List of variable for which create dummies
    Returns:
        A tuple (DF_X, DF_y) containing,
            DF_X  (pd.DataFram): dataframe with predicotrs
            DF_y  (pd.DataFram): dataframe with independent variable

    '''
    
    DF = DataDF.copy().loc[:, [DependentVar] + IndependentVar]   
    
    # CREATE DUMMY VARIABLES FOR 'DummyForCol' COLUMNS
    DF = CreateDummyForColumns(DF, DummyForCol)                    
   
    ### SPLIT INTO DEPENDENT AND INDEPANEDNT VARIABLES
    DF_y = DF.copy().loc[:, [DependentVar] ]
    DF_X = DF.copy().drop(columns= [DependentVar] )

      
    return (DF_X, DF_y)

#########################################################


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


#########################################################


def ScaleThenConvertArrayToDF(BasicDF, Scaler):
    
    BasicArray_sld = Scaler.transform(BasicDF)
    
    # convert to DataFrames becouse After scaling data stracture is array
    BasicDF_sld = pd.DataFrame(BasicArray_sld,\
                               index   = BasicDF.index,\
                               columns = BasicDF.columns)
            
    return BasicDF_sld


#########################################################


def is_list_of_strings(lst):
        return bool(lst) and isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)
        # You could break it down into `if-else` constructs to make it clearer to read.

        
#########################################################

def CreateDummyForColumns(DFwithoutDummy, DummyForColumns):
    
    DF = DFwithoutDummy.copy()
    
    # CREATE DUMMY VARIABLES FOR 'DummyForCol' COLUMNS
    if DummyForColumns is not None:
        if is_list_of_strings(DummyForColumns):
            DF = pd.get_dummies(DF, columns = DummyForColumns,\
                      prefix = [ colName + '_' for colName in DummyForColumns], drop_first=True )
        else:
            DF = pd.get_dummies(DF, columns = [DummyForColumns],\
                                prefix = [DummyForColumns + '_'], drop_first=True )
    
    return DF


#########################################################
        
def PrepareDataForRegression(DataDF, DependentVar, IndependentVar,\
                             TestSplitInd, \
                             ValSplitInd = None,\
                             DummyForCol = None,\
                             ScalerType = None,\
                             ScalerRange = (0,1)):

    # DependentVar <- str, IndependentVar <- list of str
    # ScalerType <- 'MinMax' or 'Standard'
    # ScalerRange = (0,1), or (-1,1)    - tuple
    
    DataToReturn = []
       
    # Select columns of DataFrame:    
    DF = DataDF.copy().loc[:, [DependentVar] + IndependentVar]
    
    # CREATE DUMMY VARIABLES FOR 'DummyForCol' COLUMNS
    '''
    if DummyForCol is not None:
        if is_list_of_strings(DummyForCol):
            DF = pd.get_dummies(DF, columns = DummyForCol,\
                      prefix = [ colName + '_' for colName in DummyForCol], drop_first=True )
        else:
            DF = pd.get_dummies(DF, columns = [DummyForCol],\
                                prefix = [DummyForCol + '_'], drop_first=True )
    '''

    DF = CreateDummyForColumns(DF, DummyForCol)
    
                
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



#########################################################

def MakeLagedVariableNames(LagsRangeList = None, LagsDirectList = None):

    LagedVarList = {}
    
    if (LagsRangeList):
        for Var, Lags in LagsRangeList.items():
            LagedVarList.update( { Var + '_Lag' + str(Lag+1): Lag+1\
                                    for Lag in range(Lags)} ) 
                
    if (LagsDirectList):
        for Var, Lags in LagsDirectList.items():
            LagedVarList.update( { Var + '_Lag' + str(Lag): Lag\
                                    for Lag in Lags} )
    
    return LagedVarList


##############################################################


def PrepareLags(DataFrame, LagsList):
    
    # LagsList is dictinary:
    # LagsList = {Var1_LagNr:LagNr, Var1_LagNr:LagNr}
    
    DF = DataFrame.copy()
    
    if LagsList is not None:
        for VarName, LagNr in LagsList.items():
        
            # Retrive Based Variable Name by substract '_Lag...'
            # Set up Laged Values
            VariableBase = re.sub(r'_Lag.+', '', VarName)
            DF[VarName] = DF[VariableBase].shift(LagNr)   
          
        MaxLags =  max(LagsList.values())           
        DF = DF.iloc[ MaxLags: , : ]
                                
    return DF


##############################################################

def SelectDummiesVariables(VarList, DummyForCol):
    
    if DummyForCol is not None:
        if is_list_of_strings(DummyForCol):
            
            ListOfSelectedVarList = []
            for DummyName in DummyForCol: 
                regexForDummy = re.compile(f'{DummyName}__[0-9]+')      
                ListOfSelectedVarList.append(\
                                [colName for colName in VarList\
                                         if regexForDummy.match(colName)] )
            
            DummiesForVarList = list( np.concatenate(ListOfSelectedVarList) )
            
        else:
            regexForDummy = re.compile(f'{DummyForCol}__[0-9]+')
            DummiesForVarList = [colName for colName in VarList\
                                     if regexForDummy.match(colName)]
    
    return DummiesForVarList

##############################################################

def SelectLagsVariables(VarList):
    
    regexForLags = re.compile(r'.*_Lag[0-9]+.*')
    LagsVarList = [colName for colName in VarList\
                                     if regexForLags.match(colName)]
    
    return LagsVarList


##############################################################

def KeepBasicIndeptVarAndDummies(DFwithAllVars, SelectedVarsList,\
                                 DummyForCol,\
                                 KeepBasicIndept = True,\
                                 KeepDummies = True):
    
    AllVarList = DFwithAllVars.columns.to_list()
    
    # Select Subset with lags or dummies:    
    VarsFromAllWithLags = SelectLagsVariables( AllVarList )
    VarsFromSelectedWithLags = SelectLagsVariables(  SelectedVarsList )
    DummiesFromAll = SelectDummiesVariables( AllVarList, DummyForCol )
    DummiesromSelected = SelectDummiesVariables( SelectedVarsList, DummyForCol  )


    if KeepBasicIndept and KeepDummies:       
        BasicIndept = list( set(AllVarList) - set(VarsFromAllWithLags) - set(DummiesFromAll) )
        Finall_features = BasicIndept + DummiesFromAll + VarsFromSelectedWithLags
        
    if KeepBasicIndept and not KeepDummies:
        BasicIndept = list( set(AllVarList) - set(VarsFromAllWithLags) - set(DummiesFromAll) )
        Finall_features = BasicIndept + VarsFromSelectedWithLags

    if not KeepBasicIndept and KeepDummies:
        BasicIndept = list( set(AllVarList) - set(VarsFromAllWithLags) - set(DummiesFromAll) )
        Finall_features = DummiesFromAll + VarsFromSelectedWithLags
    
    # Make proper order:
        
    Finall_features = [colName for colName in AllVarList if colName in Finall_features ]
    
    return Finall_features
