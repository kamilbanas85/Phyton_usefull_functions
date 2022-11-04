
#%% Download Data

from Download_And_Porcess_Data import Download_And_Process_Data
ResultsDIC = Download_And_Process_Data(Authenication = 'trusted', returnData = True)  
# Authenication =  'trusted'  or  'password'  or  'trusted-azure'

#%% Ectract Data <- assign variables to them names from dictinary

for key,val in ResultsDIC.items():
      exec(key + '=val')


#%% Write data to csv file

def write_dict_to_csv(df, fileName):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\assets\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/assets/dataset/" + fileName

    # write data fo csv file
    pd.DataFrame.from_dict(df, orient="index").to_csv(pathWithFileName, header=False)



def write_list_to_csv(DataList, fileName):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\assets\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/assets/dataset/" + fileName

    # write data fo csv file
    pd.DataFrame(DataList).to_csv(pathWithFileName, header=False, index=False)



def CreateMarks( DF ):
    
    MarksAll = {i:datetime.datetime.strftime(x, "%d-%m-%Y") for i,x in enumerate(DF.index.to_list())}
    MarksMax = len(MarksAll) -1
    
    Marks = {key:value[-4:] for key, value in MarksAll.items() if datetime.datetime.strptime(value, "%d-%m-%Y").month == 1 and 
                                                                  datetime.datetime.strptime(value, "%d-%m-%Y").day == 1}
    try:
        Marks[0]
    except:
        Marks[0] = ''  
    
    
    try:
        Marks[MarksMax]
    except:
        Marks[MarksMax] = ''
    
    return Marks



def CreateAndWriteMarks(CountryShort, Con_DF, Power_DF = None, PowerYears_DF = None, includePower = True):
    
    ConHist_Marks = CreateMarks(Con_DF)
    write_dict_to_csv(ConHist_Marks, f"ConHist_Marks_{CountryShort}.csv")
   
    
    if includePower:
        
        Power_Marks = CreateMarks(Power_DF)    
        PowerSources = PowerYears_DF['Sector'].unique()
    
        write_dict_to_csv(Power_Marks, f"Power_Marks_{CountryShort}.csv")  
        write_list_to_csv(PowerSources, f"PowerSources_{CountryShort}.csv")



#%% Crete Marks For Plots

CreateAndWriteMarks('EU', Con_EU, includePower = False)




########################################################

#%% Read Data From Disk


def read_Marks(fileName):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\assets\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/assets/dataset/" + fileName
    
    Marks = pd.read_csv(pathWithFileName, header=None, dtype={0: int, 1:str})\
                         .set_index(0).squeeze().to_dict()
    
    Marks = { k:'' if pd.isna(v) else v for k,v in Marks.items()  }                     
    
    return Marks


def read_list_to_data(fileName, dataType = 'int'):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\assets\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/assets/dataset/" + fileName
    
    DataList = pd.read_csv(pathWithFileName, header=None, dtype={0: dataType})\
                        .squeeze().to_list()

    return tuple(DataList)


def CreteStartAndMaxMarks(MarksDic):
    
    MarksStartValue = list({key:value for key, value in MarksDic.items() if value == str(datetime.datetime.now().year - 3)}.keys())[0]
    MarksMax = max( list(MarksDic.keys() ) )
    
    return MarksStartValue, MarksMax



ConHist_Marks = read_Marks(f"ConHist_Marks_{CountryShort}.csv")



#######################################


 dcc.RangeSlider( id=f"Slider_ConHist_{CountryShort}",
                                   marks = ConHist_Marks,
                                   min=0,
                                   max=ConHist_MarksMax,
                                   value=[ConHist_MarksStartValue, ConHist_MarksMax ]
                      )
  
 #########################################

    DF = pd.read_json(Con_from_Store).copy()*scalerUnit
        
    if Slider[1] == DF.shape[0]-1:
        DF = DF.iloc[ Slider[0] : Slider[1]+1 ]
    else:
        DF = DF.iloc[ Slider[0] : Slider[1] ]



#########################################




    
    


import dash

## add in case of pages !!!!!!!!!!!!!!!!!!!!!!
title = 'Europe-Demand'
dash.register_page(
    __name__,
    path="/",
    title=title,
    name='Europe'
)


import numpy as np
import pandas as pd
import datetime


from dash import Dash, dcc, html, Input, Output, callback, register_page

import plotly.graph_objects  as go
import plotly.express as px

import os
import pyodbc

import json

#%%

# MainDirectory = os.path.abspath(os.path.dirname(__file__))
# os.chdir(MainDirectory)


#%%

#%% 

Country = 'Europe'  
CountryShort = 'EU'


#%% Read Data From Disk


def read_csv_to_dict(fileName):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/dataset/" + fileName
    
    return pd.read_csv(pathWithFileName, header=None, dtype={0: int, 1:str})\
                        .set_index(0).squeeze().to_dict()


def read_list_to_data(fileName, dataType = 'int'):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/dataset/" + fileName
    
    DataList = pd.read_csv(pathWithFileName, header=None, dtype={0: dataType})\
                        .squeeze().to_list()

    return tuple(DataList)

   
ConHist_Marks = read_csv_to_dict(f"ConHist_Marks_{CountryShort}.csv")
ConHist_MarksMin, ConHist_MarksMax = read_list_to_data(f"ConHist_MarksMinMax_{CountryShort}.csv")


#%%

# !!! for solo page:
# dash_app = dash.Dash(__name__)
# app = dash_app.server


........



  html.Div([
       html.Div([ dcc.Graph(id=f"ConHist_plot_{CountryShort}"),
                  dcc.RangeSlider( id=f"Slider_ConHist_{CountryShort}",
                                   min=ConHist_MarksMin,
                                   max=ConHist_MarksMax,
                                   marks = ConHist_Marks,
                                   value=[list(ConHist_Marks.keys())[-4], ConHist_MarksMax ]
                      )
                ],
                style={'width': '90%',  'display': 'inline-block', 'vertical-align': 'top'} ),
       
       html.Div([  
         
         
         
   ........
         
          ], style = {'margin-bottom': '60px'}),
       

   
  ], style={'margin-right':'2%',
            'margin-left':'2%',
            'margin-top':'20px',
            'font-family': 'Times New Roman'})
          
         
#########################################################
         
  .........
         
   #############################################################################################
#### #### DEFINE Consumption-History Plot

## In case of solo page !!!!!!!!!!!!!!!!!!!!!!:
@callback(
## In case of multi-pages !!!!!!!!!!!!!!!!!!!!!!:
# @dash_app.callback(
    Output(f"ConHist_plot_{CountryShort}", 'figure'),
    [Input(f"store_Con_{CountryShort}", "data"),
     Input(f"SelectSector_ConHist_button_{CountryShort}", 'value'),
     Input(f"Freq_ConHist_button_{CountryShort}", 'value'),
     Input(f"Unit_ConHist_button_{CountryShort}", 'value'),
     Input(f"Slider_ConHist_{CountryShort}", 'value')] )
def update_Consumption_figure(Con_from_Store
                              ,Sector
                              ,Freq
                              ,Unit
                              ,Slider):

    
    colorList = px.colors.qualitative.Dark24

    
    figCon = go.Figure()
    AxisFont = 16
    AxisTitleFont = 16
    LegendFontSize = 16
    TitleFontSize = 32
    
        
    Add_Forecast_Curve = False        
        
    scalerUnit = 1    
    if Unit == 'TWh':
            scalerUnit = ConvertGWhToTWh
    elif Unit == 'BCM':
            scalerUnit = ConvertGWhToBcm
    
    DF = ( pd.read_json(Con_from_Store).copy()*scalerUnit )\
                 .loc[ datetime.datetime.fromtimestamp(Slider[0]).date() : datetime.datetime.fromtimestamp(Slider[1]).date() ]
         
         
         
    ........
