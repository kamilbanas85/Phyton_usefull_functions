import numpy as np
import pandas as pd


##############################################

def MAE(y, yhat):
    
    yhatV = yhat.values.reshape(-1)
    yV = y.values.reshape(-1)
    
    MAE = np.mean( np.absolute(yV - yhatV) )
    
    return round(MAE, 2)

##############################################

def RSME(y, yhat):
    
    yhatV = yhat.values.reshape(-1)
    yV = y.values.reshape(-1)
    
    RSME = np.sqrt( np.mean( np.square(yV - yhatV) ) )
    
    return round(RSME, 2)

##############################################

def MAPE(y, yhat):
    
    yhatV = yhat.values.reshape(-1)
    yV = y.values.reshape(-1)
    
    MAPE = np.mean( np.abs((yV - yhatV)/yV) )*100
    
    return round(MAPE, 2)

##############################################


def CalculateR2andR2adj(y, yhat , X, const = True):
    
    yhatV = yhat.values.reshape(-1)
    yV = y.values.reshape(-1)
    NoOfRegressors = len(X.columns)
    if const:
        NoOfRegressors = NoOfRegressors-1
    
    SS_Residual = sum((yV - yhatV)**2)       
    SS_Total = sum(( yV - np.mean(yV) )**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    
    adjusted_r_squared = 1 - (1-r_squared)*(len(yV)-1)\
                            /( len(yV) - NoOfRegressors - 1 )
     
#    r_squared = round(r_squared[0], 4)
#    adjusted_r_squared = round(adjusted_r_squared[0], 4)
    r_squared = round(r_squared, 4)
    adjusted_r_squared = round(adjusted_r_squared, 4)
    
    print(f'R2: {r_squared}')
    print(f'R2_adj: {adjusted_r_squared}')
    
    return (r_squared, adjusted_r_squared)


##############################################


def CalculateR2andR2adjForSatModelsWithFormula(DataDF, DependedVariable,\
                                               Model):
    
    yhat = Model.predict(DataDF)
    SS_Residual = sum((DataDF[DependedVariable]-yhat)**2)       
    SS_Total = sum(( DataDF[DependedVariable]\
                            -np.mean(DataDF[DependedVariable]) )**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total

    adjusted_r_squared = 1 - (1-r_squared)*(len(DataDF[DependedVariable])-1)\
                            /( len(DataDF[DependedVariable])-Model.df_model-1 )
    
    r_squared = round(r_squared, 4)
    adjusted_r_squared = round(adjusted_r_squared, 4)                        
    
    print(f'R2: {r_squared}')
    print(f'R2_adj: {adjusted_r_squared}')
    
    return (r_squared, adjusted_r_squared)
