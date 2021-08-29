import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


##############################################################################################
###  Generally, a VIF above 10 indicates a multicollinearity

def VIF(DF):
    
    VIFresultsDF = pd.DataFrame() 
    
    # calculating VIF for each feature
    VIFresultsDF["VIF"] = [ variance_inflation_factor(DF.values, i) for i in range(len(DF.columns)) ]
    
    # set index as variables names
    VIFresultsDF.index = DF.columns
    
    return VIFresultsDF


##############################################################################################
### GVIF - when an cathegorical variables or a polynomial part are included 

def GVIF(DF, VariablesList):

    resultsLIST = []

    for varName in VariablesList:

        X1 = DF.copy().filter(regex=f'{varName}.*')
        X2 = DF.copy().drop(X1.columns, axis=1)
    
        tmp_gvif = np.linalg.det( X1.corr() ) * np.linalg.det( X2.corr() ) / np.linalg.det( DF.corr() )
                  
        tmp_gvif = pd.DataFrame([tmp_gvif], columns = ["GVIF"] )
        tmp_gvif["GVIF^(1/2Df)"] = np.power(tmp_gvif["GVIF"], 1 / (2 * len(X1.columns)))
        tmp_gvif.index = [varName]
        resultsLIST.append(tmp_gvif)  
     
    return pd.concat(resultsLIST)
