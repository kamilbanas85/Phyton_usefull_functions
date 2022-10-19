#########################################################################
### get var name

   from varname import nameof
   nameof(MyVariable)

# or

   [x for x in globals() if globals()[x] is MyVariable][0]

#########################################################################
### Assign variables to them names form dictionary:   {'VarName':Var, ....}

   for key,val in ResultsDIC['figConsumption'].items():
      exec(key + '=val')



#########################################################################
### Set up return dictinary. RetrurnList is list os string
  
   for variable in RetrurnList:
      RetrurnDIC[variable] = eval(variable)


        
#########################################################################
#########################################################################
###################################################
### Sort Columns By Last Full Year Statistics

   SortedColumnList_DF = list( DF.resample('Y').sum().sort_values(by = DF.resample('Y').sum().index[-2],axis = 1, ascending = False).columns )
   DF = DF[SortedColumnList_DF]


#########################################################################
### from groupby to dicinary

# Slice by delivery date - year contracts 
   DF_ByYearDelivery = dict(tuple(DF.groupby('Year_Delivery')))


#########################################################################
### Join columns into one and drop nan

df.apply(lambda x: ','.join(x.dropna()), axis=1)

#########################################################################
### Replace columns with other data frame basen on some columns

for Sub_project in LNG_production_capacity_Replacment['Sub-project (Oryginal)'].unique():
    
    TemporaryRawDF = LNG_production_capacity_AfterReplacement.query('`Sub-project` == @Sub_project')
    LowIndexFromRaw = TemporaryRawDF.index[0]
    HighIndexFromRaw = TemporaryRawDF.index[-1]
    
    LNG_production_capacity_AfterReplacement = pd.concat([ LNG_production_capacity_AfterReplacement.iloc[:LowIndexFromRaw,:],
                                                           LNG_production_capacity_Replacment.query('`Sub-project (Oryginal)` == @Sub_project').drop(columns = ['Sub-project (Oryginal)']),
                                                           LNG_production_capacity_AfterReplacement.iloc[HighIndexFromRaw:,:][1:] 
                                               ])
    
    LNG_production_capacity_AfterReplacement.reset_index(drop=True, inplace = True)




#########################################################################
#########################################################################
### Exctract data from Excel withou Strikethrough

import openpyxl

def ExtractExcelDataWithoutStrikethrough(FileName):
    
    wb = openpyxl.load_workbook(FileName)
    ws = wb.worksheets[0]
    rowsListvalue=[]

    for row in ws:
        rowsListvalue.append( [None if cell.font.strike is True else cell.value  for cell in row] )

    DataDF =  pd.DataFrame(rowsListvalue)
    return DataDF


#########################################################################

import pandas as pd
import datetime

import os
import openpyxl


import requests
from bs4 import BeautifulSoup
import re


MainDirectory = os.path.abspath(os.path.dirname(__file__))
os.chdir(MainDirectory)

#%%

MainURL = 'https://www.eex.com/en/markets/trading-ressources/calendar'
page = requests.get(MainURL)

soup = BeautifulSoup(page.content, 'html.parser')

TrDataList = soup.find_all("tr")
# TrDataList[0]
CurrentYear = datetime.date.today().year
        
ListOfA = []
for Tr in TrDataList:
    
    if Tr.find_all('td', {"data-field": 'title'}, re.compile("EUA Primary Auction Calendar History") ) and\
        Tr.find_all('a', href=re.compile("zip") ):
            
        ListOfA.append( Tr.find('a')['href'] )
    

urlList = ['https://www.eex.com'+ListElm for ListElm in ListOfA]

urlAuctionCalendarCurrentYear = [ListElm for ListElm in urlList if str(CurrentYear) in ListElm][0]
AuctionCalendarCurrentYearFileName = urlAuctionCalendarCurrentYear.split('/')[-1]


response = requests.get(urlAuctionCalendarCurrentYear)
if response.status_code == 200:
    with open(MainDirectory+'\\'+AuctionCalendarCurrentYearFileName, 'wb') as f:
        f.write(response.content)

#########################################################################################
#########################################################################################
#########################################################################################

# Remove multindex after pivot_table
# .pivot_table(index=['Date'],columns = 'currencyCode', values = ['midValue'])


DF(index=['dateCET'], columns = 'scenario', values='AGSI_full')\
                       .rename_axis(None, axis=1)






VarDataFrom__currency_values.T.reset_index(drop=True).T\
    .rename( columns = {0: VarDataFrom__currency_values.columns[0][1],
                        1: VarDataFrom__currency_values.columns[1][1]} )

    
#%% or 

VarDataFrom__currency_values.columns =  list( VarDataFrom__currency_values.columns.get_level_values(1).values )




pip install --user tensorflow==2.4.1


