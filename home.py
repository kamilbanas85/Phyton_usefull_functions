
dates = Con_DE.index.to_list()
mytotaldates = {i:datetime.datetime.strftime(x, "%d-%m-%Y") for i,x in enumerate(dates)}
a = (list(mytotaldates.keys()))


Hist_Marks_1d = {key:value for key, value in mytotaldates.items() if datetime.datetime.strptime(value, "%d-%m-%Y").month == 1 and 
                                                                     datetime.datetime.strptime(value, "%d-%m-%Y").day == 1}
Hist_Marks_1d[0] = ''
Hist_Marks_1d[len(dates)-1] = ''



               dcc.RangeSlider( id=f"Slider_History_1d",
                                marks = Hist_Marks_1d,
                                min=0,
                                max=len(dates)-1,
                                value=[0, len(dates)-1 ]
                      )
                ],
                style={'width': '90%',  'display': 'inline-block', 'vertical-align': 'top'} ),
      
      
      
    print(Slider)
    DF = ( pd.read_json(Data_from_Store).copy() )\
                 .iloc[ Slider[0] : Slider[1] ]
    print(DF)
    
    


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
