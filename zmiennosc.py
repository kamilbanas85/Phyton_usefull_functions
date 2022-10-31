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
#Import the required modules for test statistic calculation:
import statsmodels.stats as sm_stat

#Import the required modules for time series model estimation:
import statsmodels.tsa as smt


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



with connIntegrated:
    crs = connIntegrated.cursor()
    VarDataFrom__Long_term_modified = pd.read_sql_query( \
                          sqlQueryTakeFrom__long_term_modified, connIntegrated)
        

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

#%% Define Functions

def tsdisplay(y, figsize = (14, 8), title = "", lags = 20):
    tmp_data = pd.Series(y)
    fig = plt.figure(figsize = figsize)
    #Plot the time series
    tmp_data.plot(ax = fig.add_subplot(311), title = "$Time\ Series\ " + title + "$", legend = False)
    #Plot the ACF:
    sm.graphics.tsa.plot_acf(tmp_data, lags = lags, zero = False, ax = fig.add_subplot(323))
    plt.xticks(np.arange(1,  lags + 1, 1.0))
    #Plot the PACF:
    sm.graphics.tsa.plot_pacf(tmp_data, lags = lags, zero = False, ax = fig.add_subplot(324))
    plt.xticks(np.arange(1,  lags + 1, 1.0))
    #Plot the QQ plot of the data:
    sm.qqplot(tmp_data, line='s', ax = fig.add_subplot(325)) 
    plt.title("QQ Plot")
    #Plot the residual histogram:
    fig.add_subplot(326).hist(tmp_data, bins = 40, density = 1)
    plt.title("Histogram")
    #Fix the layout of the plots:
    plt.tight_layout()
    plt.show()



def tsdiag(y, figsize = (14,8), title = "", lags = 10):
    #The data:
    tmp_data = pd.Series(y)
    #The Ljung-Box test results for the first k lags:
    tmp_acor = list(sm_stat.diagnostic.acorr_ljungbox(tmp_data, lags = lags, boxpierce = True))
    # get the p-values
    p_vals = pd.Series(tmp_acor[1])
    #Start the index from 1 instead of 0 (because Ljung-Box test is for lag values from 1 to k)
    p_vals.index += 1
    fig = plt.figure(figsize = figsize)
    #Plot the p-values:
    p_vals.plot(ax = fig.add_subplot(313), linestyle='', marker='o', title = "p-values for Ljung-Box statistic", legend = False)
    #Add the horizontal 0.05 critical value line
    plt.axhline(y = 0.05, color = 'blue', linestyle='--')
    # Annotate the p-value points above and to the left of the vertex
    x = np.arange(p_vals.size) + 1
    for X, Y, Z in zip(x, p_vals, p_vals):
        plt.annotate(round(Z, 4), xy=(X,Y), xytext=(-5, 5), ha = 'left', textcoords='offset points')
    plt.show()
    # Return the statistics:
    col_index = ["Ljung-Box: X-squared", "Ljung-Box: p-value", "Box-Pierce: X-squared", "Box-Pierce: p-value"]
    return pd.DataFrame(tmp_acor, index = col_index, columns = range(1, len(tmp_acor[0]) + 1))



#%% PLOT basic data

# If we wanted to estimate an ARCH model on this data, we would first have to create a mean model for the returns r_t.
# Then we should inspect the residuals - if their ACF and PACF plots show no autocorrelation but
# the squared residual ACF and PACF plots - show autocorrelation -
# we should create an ARCH model for the residuals.
#
#
# But if return plot show autocorelation, it should be fitted with 
#

tsdisplay(DF['Return'], title = "of TGE spot price return")
tsdiag(DF['Return'])


tsdisplay( DF['Return']**2, title = "of TGE spot price return")
tsdiag(DF['Return'])

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#  the significant autocorrelation in squered returns indicates ARCH effects.




#%%###########################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
### GARCH - fit model on whole dataset

# dist: 'normal', 't', 'skewt'

### DEFINE MODELS:
# Specify GARCH model with t-student disribution
GarchModel = arch_model(DF['Return'], p = 1, q = 1, o = 1, power = 1.0,\
                        mean = 'constant', vol = 'GARCH', dist = 't')

#### FIT

# Fit the model
gm_result = GarchModel.fit(update_freq=5, disp = 'off')
# print(gm_result.summary())


###################################################
# CREATE DataFrame with Extracted data


Fitted__DF = DF[['Return']].rename(columns = {'Return':'r_t'})
Fitted__DF['r_t__Fitted'] =  DF['Return'] - gm_result.resid
Fitted__DF['sigma'] = gm_result.conditional_volatility
Fitted__DF['r_t_Fit__+__sigma'] = Fitted__DF['r_t__Fitted'] + 1 *Fitted__DF['sigma']
Fitted__DF['r_t_Fit__-__sigma'] = Fitted__DF['r_t__Fitted'] - 1 *Fitted__DF['sigma']
Fitted__DF['r_t_Fit__+__2_sigma'] = Fitted__DF['r_t__Fitted'] + 2 *Fitted__DF['sigma']
Fitted__DF['r_t_Fit__-__2_sigma'] = Fitted__DF['r_t__Fitted'] - 2 *Fitted__DF['sigma']


##############################################################################
##############################################################################
#%%  PLOT FITTED DATA

plt.figure()
plt.plot(Fitted__DF['r_t'], label = "r_t")
plt.plot(Fitted__DF['r_t_Fit__+__sigma'] , linestyle ="--", label = "$\widehat{r}_t +  \widehat{\sigma}_t$")
plt.plot(Fitted__DF['r_t_Fit__-__sigma'], linestyle ="--", label = "$\widehat{r}_t - \widehat{\sigma}_t$")
plt.plot(Fitted__DF['r_t_Fit__+__2_sigma'] , linestyle ="--", label = "$\widehat{r}_t + 2 \cdot \widehat{\sigma}_t$")
plt.plot(Fitted__DF['r_t_Fit__-__2_sigma'], linestyle ="--", label = "$\widehat{r}_t - 2 \cdot \widehat{\sigma}_t$")
# Custom start and end for the x-axis
#plt.xlim(360, len(r_t))
plt.xlim(datetime.date(2020, 1, 26))

# Custom start for the y-axis
#plt.ylim(-5)
plt.legend()
plt.show()



############################################################################################################
#%%  Forceast

forecast_horizon = 1


# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = forecast_horizon, reindex = False)


# Print the forecast variance
# print(gm_forecast.variance[-1:])


# CREATE DataFrame With Forecast

# Crate Forecast Date Range 
forecast_dates_gm = pd.date_range(Fitted__DF.index.tolist()[-1], periods=forecast_horizon+1, freq='d')[1:]
# Create Forecast DateFrame
Forecast_DF = pd.DataFrame( index = forecast_dates_gm)



## Add Mean     
Forecast_DF['r_t__Forecast'] = gm_forecast.mean.values.T
## Add Varaince
Forecast_DF['Forcast_Variance'] = gm_forecast.variance.values.T
Forecast_DF['Forcast_Residual_variance'] = gm_forecast.residual_variance.values.T

## Add sigma
Forecast_DF['sigma__Forecast'] = np.sqrt( Forecast_DF['Forcast_Variance'] )


Forecast_DF['r_t_For__+__sigma'] = Forecast_DF['r_t__Forecast'] + 1 *Forecast_DF['sigma__Forecast']
Forecast_DF['r_t_For__-__sigma'] = Forecast_DF['r_t__Forecast'] - 1 *Forecast_DF['sigma__Forecast']
Forecast_DF['r_t_For__+__2_sigma'] = Forecast_DF['r_t__Forecast'] + 2 *Forecast_DF['sigma__Forecast']
Forecast_DF['r_t_For__-__2_sigma'] = Forecast_DF['r_t__Forecast'] - 2 *Forecast_DF['sigma__Forecast']


Forecast_DF = Forecast_DF.round(1)



#%% PLOT FORECAST AND FITTED 


StartPlotDate = '2022-01-01'
Forecast_value = Forecast_DF["r_t_For__+__2_sigma"][0].round(1)


#fig = plt.figure()

fig, ax = plt.subplots()

plt.plot(Fitted__DF['r_t'], color = 'grey', label = "r_t")
plt.plot(Fitted__DF['r_t_Fit__+__2_sigma'], color = 'red' , linestyle ="--", label = "$\widehat{r}_t + 2 \cdot \widehat{\sigma}_t$")
plt.plot(Fitted__DF['r_t_Fit__-__2_sigma'], color = 'blue', linestyle ="--", label = "$\widehat{r}_t - 2 \cdot \widehat{\sigma}_t$")
plt.plot(Forecast_DF['r_t_For__+__2_sigma'], color = 'red', alpha = 0.4, linestyle = "--", label = "$\widehat{r}_{T+h} + 2 \widehat{\sigma}_{T+h}$")
plt.plot(Forecast_DF['r_t_For__-__2_sigma'], color = 'blue', alpha = 0.4, linestyle = "--", label = "$\widehat{r}_{T+h} - 2 \widehat{\sigma}_{T+h}$")
# plt.fill_between(Forecast_DF['r_t_For__+__2_sigma'], Forecast_DF['r_t_For__-__2_sigma'], color = "lightblue",alpha = 0.9)
#
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

plt.xlim( pd.to_datetime(StartPlotDate), Forecast_DF.index.values[-1])
plt.grid(True)

plt.text(0.70, 0.9, f'Forecast value: $ r_t +2 \dot \sigma = {Forecast_value} \% $', fontsize=20, transform=plt.gca().transAxes,
         ha='left', va='bottom')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),
            fontsize='small', ncol=1)


# Put a legend to the right of the current axis
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# Put a legend below current axis
# Shrink current axis by 20%

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])


#plt.legend(ncol = 2)
plt.title("Returns, $r_t$")

