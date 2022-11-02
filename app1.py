import dash
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
import datetime

from dash import Dash, dcc, html, Input, Output, callback

import plotly.graph_objects  as go
import plotly.express as px

import os
import pyodbc
# import feather


#%%

# MainDirectory = os.path.abspath(os.path.dirname(__file__))
# os.chdir(MainDirectory)

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
        pathWithFileName = os.getcwd() + "\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/dataset/" + fileName

    # write data fo csv file
    pd.DataFrame.from_dict(df, orient="index").to_csv(pathWithFileName, header=False)


def write_list_to_csv(DataList, fileName):
    
    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/dataset/" + fileName

    # write data fo csv file
    pd.DataFrame(DataList).to_csv(pathWithFileName, header=False, index=False)



def CreateAndWriteMarks(CountryShort, Con_DF, Power_DF = None, PowerYears_DF = None, includePower = True):
    
    ConHist_Marks = dict(zip( Con_DF.index.to_period("Y").to_timestamp().astype(np.int64) // 10**9,\
                              Con_DF.index.to_period("Y").strftime("%Y")) )


    ConHist_MarksMin =  ( Con_DF.index.astype(np.int64) // 10**9 )[0] 
    ConHist_MarksMax =  ( Con_DF.index.astype(np.int64) // 10**9 )[-1]     
    
    
    write_dict_to_csv(ConHist_Marks, f"ConHist_Marks_{CountryShort}.csv")
    write_list_to_csv([ConHist_MarksMin, ConHist_MarksMax], f"ConHist_MarksMinMax_{CountryShort}.csv")
   
    
    if includePower:
        Power_Marks = dict(zip( Power_DF.index.to_period("Y").to_timestamp().astype(np.int64) // 10**9,\
                                Power_DF.index.to_period("Y").strftime("%Y")) )
            
        Power_MarksMin =  ( Power_DF.index.astype(np.int64) // 10**9 )[0] 
        Power_MarksMax =  ( Power_DF.index.astype(np.int64) // 10**9 )[-1] 
    
        PowerSources = PowerYears_DF['Sector'].unique()

    
        write_dict_to_csv(Power_Marks, f"Power_Marks_{CountryShort}.csv")  
        write_list_to_csv([Power_MarksMin, Power_MarksMax], f"Power_MarksMinMax_{CountryShort}.csv")
        write_list_to_csv(PowerSources, f"PowerSources_{CountryShort}.csv")


#%% Crete Marks For Plots

CreateAndWriteMarks('EU', Con_EU, includePower = False)

# Germany
CreateAndWriteMarks('DE', Con_DE, Power_DE, PowerYears_DE)
# UK
CreateAndWriteMarks('UK', Con_UK, Power_UK, PowerYears_UK)
# France
CreateAndWriteMarks('FR', Con_FR, Power_FR, PowerYears_FR)
# Italy
CreateAndWriteMarks('IT', Con_IT, Power_IT, PowerYears_IT)
# Netherlands
CreateAndWriteMarks('NL', Con_NL, Power_NL, PowerYears_NL)
# Spain
CreateAndWriteMarks('ES', Con_ES, Power_ES, PowerYears_ES)
# Poland
CreateAndWriteMarks('PL', Con_PL, Power_PL, PowerYears_PL)
# Belgium
CreateAndWriteMarks('BE', Con_BE, Power_BE, PowerYears_BE)




#%%

# LayoutRefreshTimeInMin = 120


dash_app  = dash.Dash(__name__,\
                      use_pages=True,\
                      external_stylesheets=[dbc.themes.BOOTSTRAP])

app = dash_app.server    

# app.layout = html.Div([
    
#     dcc.Store(id="store_Demand_01", data = Demand_Electricity_Texas.to_json() ),
#     dcc.Store(id="store_Demand_02", data = Demand_Electricity_TexasForecast.to_json() ),

#  	html.H1('Multi-page app with Dash Pages'),

#     html.Div(
#         [
#             html.Div(
#                 dcc.Link(
#                     f"{page['name']} - {page['path']}", href=page["relative_path"]
#                 )
#             )
#             for page in dash.page_registry.values()
#         ]
#     ),

#  	dash.page_container
# ])

navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="Demand-Country",
        toggle_style={"color": "white",  "backgroundColor":"blue"}
        
    ),
    
    brand="SUPPLY-DEMAND BALANCE",
    color="primary",
    dark=True,
    # className="mb-2",
    fluid = True,
    #style={'font-family': 'Times New Roman'}
)

dash_app.layout = dbc.Container([
    
    dcc.Store(id="store_Con_EU", data = Con_EU.to_json() ),
    dcc.Store(id="store_ConGasYears_EU", data = ConGasYears_EU.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_EU", data = ConYears_EU.reset_index().to_json() ),
    
    # Germany
    dcc.Store(id="store_Con_DE", data = Con_DE.to_json() ),
    dcc.Store(id="store_ConGasYears_DE", data = ConGasYears_DE.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_DE", data = ConYears_DE.reset_index().to_json() ),
    dcc.Store(id="store_Power_DE", data = Power_DE.to_json() ),
    dcc.Store(id="store_PowerYears_DE", data = PowerYears_DE.reset_index().to_json() ),

    # UK
    dcc.Store(id="store_Con_UK", data = Con_UK.to_json() ),
    dcc.Store(id="store_ConGasYears_UK", data = ConGasYears_UK.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_UK", data = ConYears_UK.reset_index().to_json() ),
    dcc.Store(id="store_Power_UK", data = Power_UK.to_json() ),
    dcc.Store(id="store_PowerYears_UK", data = PowerYears_UK.reset_index().to_json() ),
    
    # France
    dcc.Store(id="store_Con_FR", data = Con_FR.to_json() ),
    dcc.Store(id="store_ConGasYears_FR", data = ConGasYears_FR.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_FR", data = ConYears_FR.reset_index().to_json() ),
    dcc.Store(id="store_Power_FR", data = Power_FR.to_json() ),
    dcc.Store(id="store_PowerYears_FR", data = PowerYears_FR.reset_index().to_json() ),
    
    # Italy
    dcc.Store(id="store_Con_IT", data = Con_IT.to_json() ),
    dcc.Store(id="store_ConGasYears_IT", data = ConGasYears_IT.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_IT", data = ConYears_IT.reset_index().to_json() ),
    dcc.Store(id="store_Power_IT", data = Power_IT.to_json() ),
    dcc.Store(id="store_PowerYears_IT", data = PowerYears_IT.reset_index().to_json() ),
    
    # Netherlands
    dcc.Store(id="store_Con_NL", data = Con_NL.to_json() ),
    dcc.Store(id="store_ConGasYears_NL", data = ConGasYears_NL.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_NL", data = ConYears_NL.reset_index().to_json() ),
    dcc.Store(id="store_Power_NL", data = Power_NL.to_json() ),
    dcc.Store(id="store_PowerYears_NL", data = PowerYears_NL.reset_index().to_json() ),    

    # Spain
    dcc.Store(id="store_Con_ES", data = Con_ES.to_json() ),
    dcc.Store(id="store_ConGasYears_ES", data = ConGasYears_ES.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_ES", data = ConYears_ES.reset_index().to_json() ),
    dcc.Store(id="store_Power_ES", data = Power_ES.to_json() ),
    dcc.Store(id="store_PowerYears_ES", data = PowerYears_ES.reset_index().to_json() ),  

    # Poland
    dcc.Store(id="store_Con_PL", data = Con_PL.to_json() ),
    dcc.Store(id="store_ConGasYears_PL", data = ConGasYears_PL.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_PL", data = ConYears_PL.reset_index().to_json() ),
    dcc.Store(id="store_Power_PL", data = Power_PL.to_json() ),
    dcc.Store(id="store_PowerYears_PL", data = PowerYears_PL.reset_index().to_json() ),
    
    # Belgium
    dcc.Store(id="store_Con_BE", data = Con_BE.to_json() ),
    dcc.Store(id="store_ConGasYears_BE", data = ConGasYears_BE.reset_index().to_json() ),
    dcc.Store(id="store_ConYears_BE", data = ConYears_BE.reset_index().to_json() ),
    dcc.Store(id="store_Power_BE", data = Power_BE.to_json() ),
    dcc.Store(id="store_PowerYears_BE", data = PowerYears_BE.reset_index().to_json() ),


    navbar, 
    dash.page_container
    
    ],
    fluid=True,
)

######################################3

if __name__ == '__main__':
    
    
    dash_app.run_server(debug=True)
