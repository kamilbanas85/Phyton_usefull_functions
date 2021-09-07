import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns


# https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
##############################################################################################
#  Looking at pairs of correlations does not help us to establish
#  whether more than two variables have a linear correlation.
#  What the VIF test reveals is how much the coefficient errors
#  “grow” when the rest of the variables are present
##############################################################################################



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

##############################################################################################
### Correlation Maps based on: https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e
'''
# basic
sns.heatmap( AnalysisData.corr() )

# with values
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(AnalysisData.corr(), vmin=-1, vmax=1, annot=True)# Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
'''

##############################################################################################
### Plot colrrelations Map with values and colors with divergence map

def PlotCorrelationMap(DF):
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(DF.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
 

##############################################################################################
### Plot only correlations with dependent variable

def PlotCorrelationMapRelativeToVariable(DF, varName):
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(DF.corr()[[varName]].drop(varName, axis = 0)\
                          .sort_values(by= varName , ascending=False),\
                          vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title(f'Features Correlating with {varName}', fontdict={'fontsize':18}, pad=16)
