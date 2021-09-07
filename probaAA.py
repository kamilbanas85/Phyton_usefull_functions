import numpy as np
import pandas as pd
import os
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm

MainDirectory = os.path.abspath(os.path.dirname(__file__))
os.chdir(MainDirectory)

#%% Function to extract code from GitHub

def GetGitHubCode(GitUrl):

    response = requests.get(GitUrl) #get data from json file located at specified URL 

    if response.status_code == requests.codes.ok:
        contentOfUrl = response.content
        exec(contentOfUrl, globals() )
    else:
        print('Content was not found.')

#%% Download functions from GitHub:

GitUrl__Prepare_Data_For_Regression = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Prepare_Data_For_Regression.py'
GetGitHubCode(GitUrl__Prepare_Data_For_Regression)

GitUrl__Make_TS_Regression = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Make_TS_Regression.py'
GetGitHubCode(GitUrl__Make_TS_Regression)

GitUrl__Goodness_Of_Fit = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Goodness_Of_Fit.py'
GetGitHubCode(GitUrl__Goodness_Of_Fit)

GitUrl__Multicollinearity_Check_Functions = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Multicollinearity_Check_Functions.py'
GetGitHubCode(GitUrl__Multicollinearity_Check_Functions)

GitUrl__Feature_Selection = 'https://raw.githubusercontent.com/kamilbanas85/Phyton_usefull_functions/main/Feature_Selection.py'
GetGitHubCode(GitUrl__Feature_Selection)

#################################################################
#################################################################
#################################################################
#%% Downolad AnalysisData from ther script:

from Prepare_Electricty_Demand_Data import AnalysisData


#################################################################
#################################################################
#################################################################
#%%  Select Main Data

# AnalysisData.columns


Independent_Vars = ['HDD',
				   'CDD',
                   'wind_speed',
                   'wind_direction',
                   'humidity',
                   'sky_cover',
                   'Population',
#                   'RealGDP',
                   'WorkDay',
                   'hour',
#                   'day',
#                   'week'
                   'month'
                   ]
				   
Dependent_Var = 'Demand'
DummyForColumn = ['hour','month']
ValSetDate = '2019-06'
TestSetDate = '2021-01'

LagList = None

############################################################################################################
############################################################################################################
#%% Check Multicollinearity: VIF test and Correlations
############################################################################################################
# first look at CORELLATIONS plots:

PlotCorrelationMap(AnalysisData[Independent_Vars])

PlotCorrelationMapRelativeToVariable(AnalysisData[Independent_Vars + [Dependent_Var] ], Dependent_Var)

# AnalysisData[['Population','RealGDP']].plot()
# AnalysisData[['Population','RealGDP']].corr()

###  check VIF or GVIF (set dummy variables) 
VIF( CreateDummyForColumns( AnalysisData[Independent_Vars], DummyForColumn ) )

# Becouse data contain dummy variables, so GVIF test sholud be made
GVIF( CreateDummyForColumns( AnalysisData[Independent_Vars], DummyForColumn ),\
     Independent_Vars)

### GVIF showed that is not problem with multiollinearity,
### However correaltion between varaibles: 'Population' and 'RealGDP'
### is high ( around 0.93 ) so, one of them will be droped.
###  
### 'humidity' and 'Population'removed due to mulicollinearity
######################################################################################################
##### End Multicollinearity Check
######################################################################################################

#%% Add Lags
######################################################
######################################################
######################################################

LagWindow = 48
LagsRengeList ={'Demand':48,
				'HDD':LagWindow,
				'CDD':LagWindow}
				
LagList = MakeLagedVariableNames(LagsRangeList= LagsRengeList,\
								 LagsDirectList = None)
								
AnalysisData = PrepareLags(AnalysisData, LagList)
Independent_Vars.extend( LagList.keys() )

############################################################################################################
#%% Prepare Data For Linear Regression

X_Train, y_Train,\
X_Test, y_Test=\
	PrepareDataForRegression(AnalysisData,\
							 DependentVar = Dependent_Var,\
							 IndependentVar = Independent_Vars,\
							 DummyForCol = DummyForColumn,\
							 TestSplitInd = TestSetDate,\
							 ValSplitInd = None,
							 ScalerType = None,
							 ScalerRange = None)



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!						 
#%% Learn on whole data - use Croess-Validation

X_Train = X_Train.append(X_Test)
y_Train = y_Train.append(y_Test)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!

#%% Selecting Futures - Lags:

selected_features_BE = BackwardEliminationPvalue(X_Train = X_Train, y_Train = y_Train)

##################### 1 - method - backward elimination:
selected_features_BE = BackwardEliminationPvalue(\
                                    X_Train = X_Train, y_Train = y_Train)
  
    
##################### 2 - method - stepwise-bidirectional elimination:
selected_features_BE = stepwise_selection(X_Train, y_Train)    
      

#################### 3 - method - stepwise-bidirectional backward elimination:
selected_features_BE = BidirectionalStepwiseSelection(X_Train, y_Train, elimination_criteria = "adjr2")[0]   



#################### 4 - method - Genetive Algoritms:
# kaggle.com/azimulh/feature-selection-using-evolutionaryfs-library
# https://www.kaggle.com/azimulh/feature-selection-using-evolutionaryfs-library
from EvolutionaryFS import GeneticAlgorithmFS
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data_dict={0:{'x_train':X_Train,'y_train':y_Train,'x_test':X_Test,'y_test':y_Test}}
columns_list=list(X_Train.columns)

model_object=LinearRegression(n_jobs=-1)
# model_object=LinearRegression(n_jobs=-1)
evoObj=GeneticAlgorithmFS(model=model_object,data_dict=data_dict,\
                          cost_function=mean_squared_error,\
                          average='',\
                          cost_function_improvement='decrease',\
                          columns_list=columns_list,\
                          generations=100,\
                          population=50,\
                          prob_crossover=0.9,\
                          prob_mutation=0.1,\
                          run_time=60000)
selected_features_BE__finall=evoObj.GetBestFeatures()    

#################### 5 - method - Recursive Feature Elimination
# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()


#no of features
nof_list=np.arange(1, X_Train.shape[1] )            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_Train, y_Train)
    X_test_rfe = rfe.transform(X_Test)
    model.fit(X_train_rfe, y_Train)
    score = model.score(X_test_rfe, y_Test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]

print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


cols = list(X_Train.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X_Train,y_Train)  
#Fitting the data to model
model.fit(X_rfe,y_Train)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


#%% Add based Vriables and Dummies Variables

selected_features_BE__finall = KeepBasicIndeptVarAndDummies(X_Train, \
                                                            selected_features_BE,\
                                                            DummyForColumn,\
                                                            KeepBasicIndept = True,\
                                                            KeepDummies = True)

#%% Update X set and LagList


X_Train = X_Train.filter(items= selected_features_BE__finall)
X_Test = X_Test.filter(items= selected_features_BE__finall)

LagList = { var : LagNr for var, LagNr in LagList.items() if var in selected_features_BE__finall }


####################################################################################################
####################################################################################################
### Make Model
#%% Linear Regression - ADL

AddIntercept = True

if AddIntercept:
	ModelLR = sm.OLS( y_Train, sm.add_constant(X_Train) )
else:
	ModelLR = sm.OLS( y_Train, X_Train.astype(float) )

# X_Train.info()

ModelLRFitted = ModelLR.fit()


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#%% 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''	

#%% Make Cross Validation

ErrorStat = CVtimeSeries(X_Train, y_Train,\
						Model = sm.OLS,
						DependentVar = Dependent_Var,
						Intercept = AddIntercept,
						LagsList = LagList,
						Scaler_y = None, Scaler_X = None)
						
print( np.mean(ErrorStat['MAE']).round(2) )

#%% Make Standard forecast with test set - without CV
'''
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

####################################################################################################
####################################################################################################
####################################################################################################
#%% Check Model On Test Set

yhat_Test_LR, X_Test_withLags = \
	MakeTSforecast(X_Test, ModelLRFitted,\
				   DependentVar = Dependent_Var,
				   Intecept = AddIntercept,
				   LagsList = LagList,
				   Scaler_y = None, Scaler_X = None,
				   Test_or_Forecast = 'Test')
		   
#%% plot Fitted Data
yhat_Test_LR.join(y_Test).plot()

ModelLRFitted.fittedvalues.rename('Fitted-Train').to_frame()\
		.join(y_Train)\
		.append( yhat_Test_LR.join(y_Test) )\
		.plot()

		
CalculateR2andR2adj(y_Test, yhat_Test_LR, X_Test, const = AddIntercept)
MAE(y_Test, yhat_Test_LR)
MAPE(y_Test, yhat_Test_LR)
RSME(y_Test, yhat_Test_LR)

#%% FORECATS
'''
yhat_Forecast_LR, X_Forecast_withLags = \
	MakeTSforecast(X_Forecast, ModelLRFitted,\
				   DependentVar = Dependent_Var,
				   Intercept = AddIntercept,
				   LagsList = LagList,
				   Scaler_y = None, Scaler_X = None,
				   Test_or_Forecast = 'Forecast')
'''


######################################################################
######################################################################
######################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#%% Check model residuals and OLS assumption - on Training SET

ModelLRFitted.summary()

# Plot fitted model
ModelLRFitted.fittedvalues.rename('Fitted-Train').to_frame()\
	.join(y_Train)\
    .loc['2020':]\
	.plot()

####################################################################################################
# Plot Resudulas
ModelLRFitted.resid.loc['2020-11':].plot()

####################################################################################################
####################################################################################################
#%% Check AutoCorrelation of residuals

# plot autocorelogram
fig, ax = plt.subplots(2,1, figsize=(15,8))
fig = sm.graphics.tsa.plot_acf(ModelLRFitted.resid, lags=80, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(ModelLRFitted.resid, lags=80, ax=ax[1])
plt.show()


# Durbin-Whatson Test - 1-order correlations
# A value:
#   - 2.0:               no autocorrelation
#   - 0 to less than 2:  positive autocorrelation
#   - from 2 to 4:       negative autocorrelation

from statsmodels.stats.stattools import durbin_watson
durbin_watson(ModelLRFitted.resid).round(3)
# Positive autocorelation of 1-order

# To check higher order correaltions: Ljung-box test 
from statsmodels.stats.diagnostic import acorr_ljungbox
acorr_ljungbox(ModelLRFitted.resid)
# H0 - a residuals are independently distributed
# p-values less than 0.05, so H0 is receted for a lot of lags
# so a higer oreder autocorrelations exists

####################################################################################################
####################################################################################################
#%% Check Heteroskedasticity

'''
from statsmodels.stats.diagnostic import het_white
het_white(ModelLRFitted.resid, ModelLRFitted.model.exog)
'''
# https://www.statology.org/breusch-pagan-test-python/
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan
test = het_breuschpagan(ModelLRFitted.resid,ModelLRFitted.model.exog)

names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']

lzip(names, test)
# H0: Homoscedasticity is present
# the null hypothesis is rejected if p<0.05, so in this case 
# we reject H0, so Hetercedasity is present


# Goldfeld-Quandt test
from statsmodels.stats.diagnostic import het_goldfeldquandt

name = ['F statistic', 'p-value']
test = het_goldfeldquandt(ModelLRFitted.resid, ModelLRFitted.model.exog)
lzip(name, test)
# H0 for the GQ test is homoskedasticity

####################################################################################################
####################################################################################################
#%% Correaltion Between Regressors and Residuals
## The null hypothesis is that the two variables are uncorrelated
## so p-value greater than 0.05 support uncorrelation
from scipy.stats.stats import pearsonr

resultsLIST = []
for columnName in X_Train.columns:

    CorrRegressVsResidDF_temp =\
        pd.DataFrame([pearsonr( ModelLRFitted.resid, X_Train.loc[:,columnName] )[1].round(4)],\
                     columns = ["p-value"] )
    CorrRegressVsResidDF_temp.index = [columnName]
    resultsLIST.append(CorrRegressVsResidDF_temp)  

pd.concat(resultsLIST)

####################################################################################################
####################################################################################################
#%%######################################
### Normality of residuals
# QQ-Plot of residuals
fig1 = sm.qqplot(ModelLRFitted.resid, fit=True,line='45')
plt.show()

#%% ## Linearity
#
# Harvey-Collier multiplier test for Null hypothesis that the linear
# specification is correct:

name = ['t value', 'p value']
test = sms.linear_harvey_collier(results)
lzip(name, test)

####################################################################################################
### End Of OLS assumption check
####################################################################################################

####################################################################################################
#%% Check infuence of veraibles - standarized coefficients (beta coefficients):
### it should be remembered that CONTANT is USED

###########################################
# post-processing - using current model
b = np.array( ModelLRFitted.params )[1:]
std_y = ModelLR.endog.std(axis=0)
std_x = ModelLR.exog.std(axis=0)[1:]
beta = b*(std_x/std_y)

BetaCoeffients01 = pd.DataFrame(beta , index = X_Train.columns )

###########################################
# pre-processing - using current model
from scipy import stats

ModelLRFitted_std = sm.OLS( y_Train.apply( stats.zscore), \
                            sm.add_constant(X_Train.apply( stats.zscore )) ).fit()
ModelLRFitted_std.summary()
BetaCoeffients02 = ModelLRFitted_std.params[1:]



      

