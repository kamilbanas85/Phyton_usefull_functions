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
                                DummyForCol = None,
                                drop_first=True):
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
    DF = CreateDummyForColumns(DF, DummyForCol, drop_first = drop_first)                    
   
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

def CreateDummyForColumns(DFwithoutDummy, DummyForColumns, drop_first=True):
    
    DF = DFwithoutDummy.copy()
    
    # CREATE DUMMY VARIABLES FOR 'DummyForCol' COLUMNS
    if DummyForColumns is not None:
        if is_list_of_strings(DummyForColumns):
            DF = pd.get_dummies(DF, columns = DummyForColumns,\
                      prefix = [ colName + '_' for colName in DummyForColumns], drop_first=drop_first, dtype=int)
        else:
            DF = pd.get_dummies(DF, columns = [DummyForColumns],\
                                prefix = [DummyForColumns + '_'], drop_first=drop_first, dtype=int)
    
    return DF


#########################################################
        
def PrepareDataForRegressionOLD(DataDF, DependentVar, IndependentVar,\
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

#########################################################

def make_laged_or_forwad_variable_names(LagsRangeList = None,\
                                        LagsDirectList = None,\
                                        lag_or_forward = 'lag'):

    LagedVarList = {}
    suffix = '_Lag'
    if lag_or_forward == 'lag':
        suffix = '_Lag'
    elif lag_or_forward == 'forward':
        suffix = '_Forward'
        
    
    if (LagsRangeList):
        for Var, Lags in LagsRangeList.items():
            LagedVarList.update( { Var + suffix + str(Lag+1): Lag+1\
                                    for Lag in range(Lags)} ) 
                
    if (LagsDirectList):
        for Var, Lags in LagsDirectList.items():
            LagedVarList.update( { Var + suffix + str(Lag): Lag\
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


def prepare_forwards_shift(DataFrame, forward_list):
    
    # LagsList is dictinary:
    # LagsList = {Var1_LagNr:LagNr, Var1_LagNr:LagNr}
    
    DF = DataFrame.copy()
    
    if forward_list is not None:
        for VarName, LagNr in forward_list.items():
        
            # Retrive Based Variable Name by substract '_Lag...'
            # Set up Laged Values
            VariableBase = re.sub(r'_Forward.+', '', VarName)
            DF[VarName] = DF[VariableBase].shift(-LagNr)   
          
        #MaxLags =  max(forward_list.values())           
        #DF = DF.iloc[ MaxLags: , : ]
                                
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
    DummiesFromAll = []
    
    # Select Subset with lags or dummies:    
    VarsFromAllWithLags = SelectLagsVariables( AllVarList )
    VarsFromSelectedWithLags = SelectLagsVariables(  SelectedVarsList )
    if DummyForCol is not None:
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

  
########################################################################

def TrainValTestSplitIndexIncludingBatch(DF,\
                                         TestSplitInd,\
                                         ValSplitInd,\
                                         BatchSize = None,\
                                         WindowLength = 1):
    '''
    Function return dates of Validation and Test Sets for selected Batch size.
    Actualy it return values closest to selected values.
    
    Function calculate length of the DataFrame including future cut by WindowLength.
    If data length is not divided by batch size function cut the data.
    Then function prepare length of Train set based on specified date. However, if length of 
    train set is not divided by batch size, than data it cuted from the end of set.
    
    WindowLength = 1 - means no window

    Args:
        DF (pd.DataFrame): Oryginal DataFrame with all observations
        BatchSize (int): batch size
        TestSplitInd (str): String representing date of Start of Test Set
        ValSplitInd (str): String representing date of Start of Test Set
        WindowLength (int): length of lags widnow

    Returns:
        (Train.index, Val.index, Test.index) (tuple(indexdatetime, indexdatetime, indexdatetime)):
                3 indexs representing train, validaton and test sets
            
    '''
     
    DFfinall = DF.copy()
    TotalLength = DF.shape[0]
    
    # if WindowLength > 1 cut some data from the begining to enable window creation
    if WindowLength > 1:
        DFfinall = DFfinall.iloc[(WindowLength-1):]
        TotalLength = DFfinall.shape[0]
    
    if BatchSize is None:
        Train, Test = TrainTestSets(DFfinall, TestSplitInd)
        if ValSplitInd is not None:
            Train, Val  = TrainTestSets(Train, ValSplitInd)
            return (Train.index, Val.index, Test.index)
        else:
            return (Train.index, None, Test.index)
    
    elif BatchSize is not None:    
        if TotalLength % BatchSize != 0:
#           print('Batch Size should be divisor of the data length')
            DFfinall = DFfinall.iloc[(TotalLength%BatchSize):]
        
        Train, RestDF  = TrainTestSets(DFfinall, ValSplitInd)
        if Train.shape[0] % BatchSize != 0:      
            Train = Train.iloc[:-(Train.shape[0] % BatchSize)]
            RestDF = DFfinall[Train.index[-1]:].iloc[1:]
    
        Val, Test = TrainTestSets(RestDF, TestSplitInd)
        if Val.shape[0] % BatchSize != 0:    
            Val = Val.iloc[:-(Val.shape[0] % BatchSize)]
            Test = RestDF[Val.index[-1]:].iloc[1:]

        return (Train.index, Val.index, Test.index)


################################################################################################

def CreateDFwithWindowsForLSTM(DF, IndexRange, WindowLength):
    
    '''
    Function creates DataFrame with laged windows. Ready for reshape for LSTM

    Args:
        DF (pd.DataFrame): Oryginal DataFrame with all observations
        IndexRange (Index): Selected Index Range for which create data
        WindowLength (int): length of lags widnow

    Returns:
        DFwithWindow (pd.DataFrame): DataFrame with added windowed rows
    '''
    
    DFwithWindow_List = []

    for index in IndexRange:     
        DFwithWindow_List.append( DF.loc[:index].iloc[-WindowLength:] )
       
    DFwithWindow = pd.concat( DFwithWindow_List )

    return DFwithWindow

  
################################################################################################

def SplitDataBasedOnIndex(DFx, DFy, TrainIdx, ValIdx, TestIdx, WindowLen = None):
    '''
    Function splits DataFrames into Train, Val and Test sets based on index

    Args:
        DFx (pd.DataFrame): DataFrame with independent variables
        DFy (pd.DataFrame): DataFrame with dependent variables
        
        TrainIdx (Index): Index for Train Set
        ValIdx (Index): Index for Validation Set
        TestIdx (Index): Index for Test Set

    Returns:
        NoName (tuple): tuple of 6 DataFrames
    '''
    
    if WindowLen is None:
        TrainDFx = DFx.loc[TrainIdx]
        TestDFx  = DFx.loc[TestIdx]
        if ValIdx is not None:
            ValDFx = DFx.loc[ValIdx]
        else:
            ValDFx = None
    else:
        TestLen  = len(TestIdx)*WindowLen
        if ValIdx is not None:
            ValLen = len(ValIdx)*WindowLen
        elif ValIdx is not None:
            ValLen = 0
        TrainLen = len(TrainIdx)*WindowLen

        TestDFx  = DFx.iloc[-TestLen:,:]           
        ValDFx   = DFx.iloc[-(ValLen+TestLen):-TestLen,:]             
        TrainDFx = DFx.iloc[-(TrainLen+ValLen+TestLen):-(ValLen+TestLen),:]
        
    # Only X set contain window, so y is the same for bouth:
    TrainDFy   = DFy.loc[TrainIdx]
    TestDFy    = DFy.loc[TestIdx]
    if ValIdx is not None:
        ValDFy     = DFy.loc[ValIdx]
    else:
        ValDFy = None
    
    if ValIdx is not None:
        return (TrainDFx, TrainDFy, ValDFx, ValDFy, TestDFx, TestDFy)
    elif ValIdx is None:
        return (TrainDFx, TrainDFy, TestDFx, TestDFy)

      
################################################################################################

def PrepareDataForRegression(X_df, y_df, 
                             TestSplitInd, \
                             ValSplitInd,\
                             BatchSize = None,\
                             WindowLength = 1,\
                             ScalerType = None,\
                             ScalerRange = (0,1)):

    # ScalerType <- 'MinMax' or 'Standard'
    # ScalerRange = (0,1), or (-1,1)    - tuple
    
    DF_y = y_df.copy()
    DF_X = X_df.copy()
 
    ### Divide Index into TRAIN, VALIDATION and TEST sets    
    TrainIndex, ValIndex, TestIndex =\
        TrainValTestSplitIndexIncludingBatch(DF = DF_y,
                                             TestSplitInd = TestSplitInd,
                                             ValSplitInd = ValSplitInd,
                                             BatchSize = BatchSize,
                                             WindowLength = WindowLength)
    
    # Split and Return Data if without window and without scaling
    if WindowLength == 1 and ScalerType is None:
        return SplitDataBasedOnIndex(DF_X, DF_y, TrainIndex, ValIndex, TestIndex)
    
    # Set up scaler     
    if ScalerType is not None:
        # Make PreTrainDF to scale on all traing data including widnow
        PreTrainDF_X, PreTrainDF_y = DF_X.loc[:TrainIndex[-1]], DF_y.loc[:TrainIndex[-1]]
        # Define Scalers:        
        if ScalerType == 'MinMax':    
            scaler_y = MinMaxScaler( feature_range = ScalerRange )
            scaler_X = MinMaxScaler( feature_range = ScalerRange )        
        elif ScalerType == 'Standard':    
            scaler_y = StandardScaler()
            scaler_X = StandardScaler()
    
        # Fit scalers on Trains Sets:
        scaler_y = scaler_y.fit(PreTrainDF_y)
        scaler_X = scaler_X.fit(PreTrainDF_X)   
        
        # Scale all data
        DF_y_sld = ScaleThenConvertArrayToDF(DF_y, scaler_y)
        DF_X_sld = ScaleThenConvertArrayToDF(DF_X, scaler_X)
    
    # Split and Return Data if without window and with scaling
    if WindowLength == 1 and ScalerType is not None:
        return SplitDataBasedOnIndex(DF_X_sld, DF_y_sld, TrainIndex, ValIndex, TestIndex)\
                 +  (scaler_X, scaler_y)

    ####################################################################################
    ##### Widnow > 1   <- becouse if Window ==1 data was returned
    ####################################################################################
    
    # create windowed data
    IndexRange = list(TrainIndex) + list(ValIndex) + list(TestIndex)
    if ScalerType is not None:
        DF_X_sld_withWindow = CreateDFwithWindowsForLSTM(DF = DF_X_sld,\
                                                         IndexRange = IndexRange,\
                                                         WindowLength = WindowLength)
    else:
        DF_X_withWindow = CreateDFwithWindowsForLSTM(DF = DF_X,\
                                                     IndexRange = IndexRange,\
                                                     WindowLength = WindowLength) 
    # Split data on Train, Val and Test sets and return
    if ScalerType is not None:
         return SplitDataBasedOnIndex(DF_X_sld_withWindow, DF_y_sld, TrainIndex, ValIndex, TestIndex, WindowLength) \
                     + (scaler_X, scaler_y)
    if ScalerType is None:
         return SplitDataBasedOnIndex(DF_X_withWindow, DF_y, TrainIndex, ValIndex, TestIndex, WindowLength)
