import os 
import pyodbc as pyodbc
import pandas as pd
import numpy as np

import math

from arch import arch_model
from scipy import stats

import datetime

import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


from sklearn.metrics import mean_absolute_error, mean_squared_error


MainDirectory = os.path.abspath(os.path.dirname(__file__))
os.chdir(MainDirectory)

os.chdir(FunctionsDirectory)
from Cleaning_Data_Functions import *
from TimeSeries_Forcasting_Functions import *
from SetDesiredLevelsToTrend import *

os.chdir(MainDirectory)

#######################################################
#%% Import  Data From View


        

#%% Prepare data

DF = VarDataFrom__Long_term_modified\
                             .assign(dateCET = lambda x: x['dateCET']\
                                                    .astype('datetime64') )\
                             .dropna(subset=['dateCET'])\
                             .sort_values('dateCET')\
                             .set_index('dateCET')

DF = DF.dropna()

#%% check na

# DF.asfreq('d').isna().sum()

#%% # Calculate daily returns as percentage price changes

DF['Return'] = 100 * (DF[['Spot_POLPX']].pct_change())
DF.dropna(subset = ['Return'], inplace = True)

DF['Variance'] = DF['Return'].sub(DF['Return'].mean()).pow(2)

#%% PLOT basic data

# Price Plot
fig = plt.figure()
plt.plot(DF['Spot_POLPX'], color = 'blue', label = 'Daily Returns')
plt.xlim(DF.index[0], DF.index[-1])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.xlabel('Data', fontsize=28)
plt.ylabel('Stopa Zwrotu [%]', fontsize=28)
plt.title('Cena gazu na rynku spot TGE', fontsize=34)
plt.grid(True)
plt.show()

# Return Plot
fig = plt.figure()
plt.plot(DF['Return'], color = 'blue', label = 'Daily Returns')
plt.xlim(DF.index[0], DF.index[-1])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.xlabel('Data', fontsize=28)
plt.ylabel('Stopa Zwrotu [%]', fontsize=28)
plt.title('Stopa Zwrotu cen gazu na rynku spot TGE', fontsize=34)
plt.grid(True)
plt.show()

#%% Daily volatility as the standard deviation of price returns:

std_daily = round(DF['Return'].std(), 2)

print('Daily volatility: ', '{:.2f}%'.format(std_daily))


#%% Weekly volatility

# Convert daily volatility to weekly volatility
std_weekly = math.sqrt(5) * std_daily
print ('Weekly volatility: ', '{:.2f}%'.format(std_weekly))

#%% Monthly volatility

# Convert daily volatility to monthly volatility
std_monthly = math.sqrt(21) * std_daily
print ('Monthly volatility: ', '{:.2f}%'.format(std_monthly))

#%%#####################################################
########################################################
########################################################
#%%##################################################################
#####################################################################
#####################################################################
# Expanding rolling window


# Define Models
ModelSetUpAll = {'GARCH':  {'Name':'GARCH',
                            'vol': 'GARCH',
                            'mean':'constant',
                            'dist':'t',
                            'o':0},
                'GJR-GARCH': {'Name':'GJR-GARCH',
                                'vol': 'GARCH',
                               'mean':'constant',
                               'dist':'t',
                               'o':1},
               'EGARCH': {'Name':'EGARCH',
                          'vol': 'EGARCH',
                          'mean':'constant',
                          'dist':'t',
                          'o':1}
               }



ReturnData = DF['Return']

TestDaysNr = 30
ResultsForecast = {}


# Make forcast for all specified models:
for ModelName in ModelSetUpAll.keys():
    
    ModelSetUp = ModelSetUpAll[ModelName]
    
    forecasts = {}
    
    for i in range(TestDaysNr, 0, -1):
        
        # Specify EGARCH Model 
        Garch_Model = arch_model( ReturnData.iloc[:-i], p = 1, q = 1, o = ModelSetUp['o'],
                                  mean = ModelSetUp['mean'], vol = ModelSetUp['vol'], dist = ModelSetUp['dist'])
        
        Garch_Model_Fitted = Garch_Model.fit(disp = 'off')
        Forecast_temp = Garch_Model_Fitted.forecast(horizon=1, reindex = False).variance
        #Forecast_temp.index
        #Forecast_temp.values
        #fcast = temp.iloc[0]
        forecasts[ Forecast_temp.index[0] + datetime.timedelta(days=1)] = np.sqrt( Forecast_temp.iloc[0,0] )
        
    
    forecastsDF = pd.DataFrame.from_dict(forecasts, orient='index', columns = [f'Forecast - {ModelSetUp["Name"]}']).round(1)
    MAE = mean_absolute_error(DF.loc[forecastsDF.index].Variance, forecastsDF**2).round(1)
    MSE = mean_squared_error(DF.loc[forecastsDF.index].Variance, forecastsDF**2).round(1)
    RMSE = np.sqrt(mean_squared_error(DF.loc[forecastsDF.index].Variance, forecastsDF**2)).round(1)
    
    ResultsForecast[ModelName] = {'Forecast':forecastsDF, 'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE }
    #ResultsForecast[ModelName] = forecastsDF
    

# ResultsForecast['GARCH']

#%% plot Variance

# ModelVol = ResultsForecast['GARCH']['Forecast']

# # Plot the actual Bitcoin volatility
# plt.plot(DF.loc[ModelVol.index,'Variance'], color = 'grey', alpha = 0.4, label = 'Daily Variance')

# # Plot EGARCH  estimated volatility
# plt.plot(ModelVol**2, color = 'red', label = 'Model Variance')

# plt.legend(loc = 'upper right')
# plt.show()


#%% PLOT 

# Plot all data 

fig = plt.figure()
plt.plot(ReturnData.loc[ResultsForecast['GARCH']['Forecast'].index], alpha = 0.6, color = 'grey', label = 'Returns')
plt.plot(ResultsForecast['GARCH']['Forecast'], color = 'blue', label = 'Forecast - GARCH')
plt.plot(ResultsForecast['GJR-GARCH']['Forecast'], color = 'red', label = 'Forecast - GJR-GARCH')
plt.plot(ResultsForecast['EGARCH']['Forecast'], color = 'green', label = 'Forecast - EGARCH')

plt.xlim(ResultsForecast['GARCH']['Forecast'].index[0], ResultsForecast['GARCH']['Forecast'].index[-1])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.xlabel('Data', fontsize=28)
plt.xticks(fontsize=20, rotation=0)
plt.ylabel('Stopa Zwrotu [%]', fontsize=28)
plt.title('Stopa Zwrotu cen gazu na rynku spot TGE oraz 1-dniowa prognoza jej zmienność', fontsize=30)
plt.legend(loc = 'upper left',  fontsize=17)

plt.text(0.78, 0.88, f'MAE GARCH = {ResultsForecast["GARCH"]["MAE"]} \nMAE GJR-GARCH = {ResultsForecast["GJR-GARCH"]["MAE"]}\nMAE EGARCH = {ResultsForecast["EGARCH"]["MAE"]}', 
         fontsize=20, fontfamily='Times New Roman', color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes);

# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',  fontsize=18)
#fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(right=0.75)

# Option 3
# WX backend
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)
plt.show()

#%%###########################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
### GARCH - fit model on whole dataset

# dist: 'normal', 't', 'skewt'

### DEFINE MODELS:
# Specify GARCH model with t-student disribution
GarchModel = arch_model(DF['Return'], p = 1, q = 1,
                        mean = 'constant', vol = 'GARCH', dist = 't')

# Specify GJR-GARCH Model 
GJRgarchModel = arch_model(DF['Return'], p = 1, q = 1, o=1,
                        mean = 'constant', vol = 'GARCH', dist = 't')

# Specify EGARCH Model 
EgarchModel = arch_model(DF['Return'], p = 1, q = 1, o=1,
                        mean = 'constant', vol = 'EGARCH', dist = 't')


#### FIT

# Fit the model
gm_result = GarchModel.fit(update_freq=5, disp = 'off')
gjrgm_result = GJRgarchModel.fit(update_freq=5, disp = 'off')
egm_result = EgarchModel.fit(update_freq=5, disp = 'off')


# Get model estimated volatility
vol_fitted_garch = gm_result.conditional_volatility.rename('Fitted - GARCH')
vol_fitted_gjrgarch = gjrgm_result.conditional_volatility.rename('Fitted - GJR-GARCH')
vol_fitted_egarch = egm_result.conditional_volatility.rename('Fitted - EGARCH')

# Plot Fitted Volatility
fig = plt.figure()
plt.plot(DF['Return'], color = 'grey', label = 'Returns')
plt.plot(vol_fitted_garch, color = 'red', label = 'Fitted - GARCH')
plt.plot(vol_fitted_gjrgarch, color = 'blue', label = 'Fitted - GJR-GARCH')
plt.plot(vol_fitted_egarch, color = 'green', label = 'Fitted - EGARCH')
plt.xlim(DF.index[0], DF.index[-1])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.xlabel('Data', fontsize=28)
plt.ylabel('Stopa Zwrotu [%]', fontsize=28)
plt.title('Stopa Zwrotu cen gazu na rynku spot TGE oraz zamodelowana jej zmienność', fontsize=34)
plt.legend(loc = 'upper left')
plt.grid(True)
plt.show()



#####################################################
#%% check model/residuals

results = egm_result

# Display model fitting summary
print(results.summary())

# Plot fitted results
results.plot()
plt.show()


##############################################
### check distribution of residuals

# Obtain model estimated residuals and volatility
gm_resid = results.resid
gm_std = results.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Plot the histogram of the standardized residuals
# Plot the histogram of the standardized residuals
plt.figure(figsize=(7,4))
sns.distplot(gm_std_resid, norm_hist=True, fit=stats.norm, bins=50, color='r')
plt.legend(('normal', 'standardized residuals'))
plt.show()

################################################
###  MODEL CHECK
###  the standardized residuals should not exhibit data clustering or autocorrelations. 
## #A GARCH model should resemble a white noise process.


### acf plot
# Import the Python module

# Plot the standardized residuals
plt.plot(gm_std_resid)
plt.title('Standardized Residuals')
plt.show()

# Generate ACF plot of the standardized residuals
plot_acf(gm_std_resid, alpha = 0.05)
plt.show()

########### Ljung-Box test
# Perform the Ljung-Box test
lb_test = acorr_ljungbox(gm_std_resid , lags = 10)

# Print the p-values
#All the p-values are larger than 5% so the null hypothesis cannot be rejected. 
# In other words, no autocorrelations detected and the model is doing a decent job.


########################################################
########################################################
#%% Forecast - 

# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = 5, reindex = False)
gjrgm_forecast = gjrgm_result.forecast(horizon = 5, reindex = False)
egm_forecast = egm_result.forecast(horizon = 1, reindex = False)

# Print the forecast variance
print(gm_forecast.variance[-1:])
print(gjrgm_forecast.variance[-1:])
print(egm_forecast.variance[-1:])

forecast_dates_gm = pd.date_range(gm_forecast.variance[-1:].index[0], periods=6, freq='d')[1:]
vol_forecast_gm = pd.DataFrame(np.sqrt(gm_forecast.variance[-1:].T.values),\
                            index = forecast_dates_gm, columns = ['Forecast: GARCH'])
vol_forecast_gm = vol_forecast_gm.round(1)

forecast_dates_gjrgm = pd.date_range(gjrgm_forecast.variance[-1:].index[0], periods=6, freq='d')[1:]
vol_forecast_gjrgm = pd.DataFrame(np.sqrt(gjrgm_forecast.variance[-1:].T.values),\
                            index = forecast_dates_gjrgm, columns = ['Forecast: GJR - GARCH'])
vol_forecast_gjrgm = vol_forecast_gjrgm.round(1)

forecast_dates_egm = pd.date_range(egm_forecast.variance[-1:].index[0], periods=2, freq='d')[1:]
vol_forecast_egm = pd.DataFrame(np.sqrt(egm_forecast.variance[-1:].T.values),\
                            index = forecast_dates_egm, columns = ['Forecast: EGARCH'])
vol_forecast_egm = vol_forecast_egm.round(1)

    
#%% Plot Fitted Volatility and Forecast

fig = plt.figure()
plt.plot(DF['Return'], alpha = 0.6, color = 'grey', label = 'Returns')
plt.plot(vol_fitted_garch, color = 'red', label = 'Fitted - GARCH')
plt.plot(vol_forecast_gm, color = 'olive', label = 'Forecast - GARCH')
plt.plot(vol_fitted_gjrgarch, color = 'blue', label = 'Fitted - GJR-GARCH')
plt.plot(vol_forecast_gjrgm, color = 'purple', label = 'Forecast - GJR-GARCH')
plt.plot(vol_fitted_egarch, color = 'green', label = 'Fitted - EGARCH')
plt.plot(vol_forecast_egm, color = 'lime', label = 'Forecast - EGARCH')
plt.xlim(DF.loc['2022':].index[0], vol_forecast_gm.index[-1])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.xlabel('Data', fontsize=28)
plt.xticks(fontsize=20, rotation=0)
plt.ylabel('Stopa Zwrotu [%]', fontsize=28)
plt.title('Stopa Zwrotu cen gazu na rynku spot TGE oraz zamodelowana jej zmienność', fontsize=34)
plt.legend(loc = 'upper left',  fontsize=17)
plt.text(0.85, 0.88, f'GARCH = {vol_forecast_gm.iloc[0,0]} \nGJR-GARCH = {vol_forecast_gjrgm.iloc[0,0]}\nEGARCH = {vol_forecast_egm.iloc[0,0]}', 
         fontsize=20, fontfamily='Times New Roman', color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes);
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',  fontsize=18)
#fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(right=0.75)

# Option 3
# WX backend
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)
plt.show()


########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
#%% Fixed rolling window


# Define Models
ModelSetUpAll_fixed = {'GARCH':  {'Name':'GARCH',
                            'vol': 'GARCH',
                            'mean':'constant',
                            'dist':'t',
                            'o':0},
                'GJR-GARCH': {'Name':'GJR-GARCH',
                                'vol': 'GARCH',
                               'mean':'constant',
                               'dist':'t',
                               'o':1},
               # 'EGARCH': {'Name':'EGARCH',
               #            'vol': 'EGARCH',
               #            'mean':'constant',
               #            'dist':'t',
               #            'o':1}
               }



ReturnData = DF['Return']

TrainDaysNr = 60
TestDaysNr = 60
ResultsForecast_Fixed = {}


# Make forcast for all specified models:
for ModelName in ModelSetUpAll_fixed.keys():
    
    ModelSetUp = ModelSetUpAll_fixed[ModelName]
    
    forecasts = {}
    
    for i in range(TestDaysNr, 0, -1):
        
        # Specify EGARCH Model 
        Garch_Model = arch_model( ReturnsData.iloc[-(TrainDaysNr+i):-i], p = 1, q = 1, o = ModelSetUp['o'],
                                  mean = ModelSetUp['mean'], vol = ModelSetUp['vol'], dist = ModelSetUp['dist'])
        
        Garch_Model_Fitted = Garch_Model.fit(disp = 'off')
        Forecast_temp = Garch_Model_Fitted.forecast(horizon=1, reindex = False).variance
        #Forecast_temp.index
        #Forecast_temp.values
        #fcast = temp.iloc[0]
        forecasts[ Forecast_temp.index[0] + datetime.timedelta(days=1)] = np.sqrt( Forecast_temp.iloc[0,0] )
        
    
    forecastsDF = pd.DataFrame.from_dict(forecasts, orient='index', columns = [f'Forecast - {ModelSetUp["Name"]}']).round(1)
    MAE = mean_absolute_error(DF.loc[forecastsDF.index].Variance, forecastsDF**2).round(1)
    MSE = mean_squared_error(DF.loc[forecastsDF.index].Variance, forecastsDF**2).round(1)
    RMSE = np.sqrt(mean_squared_error(DF.loc[forecastsDF.index].Variance, forecastsDF**2)).round(1)
    
    ResultsForecast_Fixed[ModelName] = {'Forecast':forecastsDF, 'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE }
    #ResultsForecast[ModelName] = forecastsDF
    

# ResultsForecast_Fixed['GARCH']

#%% plot Variance

# ModelVol = ResultsForecast['GARCH']['Forecast']

# # Plot the actual Bitcoin volatility
# plt.plot(DF.loc[ModelVol.index,'Variance'], color = 'grey', alpha = 0.4, label = 'Daily Variance')

# # Plot EGARCH  estimated volatility
# plt.plot(ModelVol**2, color = 'red', label = 'Model Variance')

# plt.legend(loc = 'upper right')
# plt.show()


#%% PLOT 

# Plot all data 

fig = plt.figure()
plt.plot(ReturnData.loc[ResultsForecast_Fixed['GARCH']['Forecast'].index], alpha = 0.6, color = 'grey', label = 'Returns')
plt.plot(ResultsForecast_Fixed['GARCH']['Forecast'], color = 'blue', label = 'Forecast - GARCH')
plt.plot(ResultsForecast_Fixed['GJR-GARCH']['Forecast'], color = 'red', label = 'Forecast - GJR-GARCH')
# plt.plot(ResultsForecast['EGARCH']['Forecast'], color = 'green', label = 'Forecast - EGARCH')

plt.xlim(ResultsForecast_Fixed['GARCH']['Forecast'].index[0], ResultsForecast_Fixed['GARCH']['Forecast'].index[-1])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.xlabel('Data', fontsize=28)
plt.xticks(fontsize=20, rotation=0)
plt.ylabel('Stopa Zwrotu [%]', fontsize=28)
plt.title('Stopa Zwrotu cen gazu na rynku spot TGE oraz 1-dniowa prognoza jej zmienność', fontsize=30)
plt.legend(loc = 'upper left',  fontsize=17)

plt.text(0.78, 0.88, f'MAE GARCH = {ResultsForecast_Fixed["GARCH"]["MAE"]} \nMAE GJR-GARCH = {ResultsForecast_Fixed["GJR-GARCH"]["MAE"]}', 
         fontsize=20, fontfamily='Times New Roman', color='k',
         ha='left', va='bottom',
         transform=plt.gca().transAxes);

# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',  fontsize=18)
#fig.subplots_adjust(bottom=0.2)
#fig.subplots_adjust(right=0.75)

# Option 3
# WX backend
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.grid(True)
plt.show()
