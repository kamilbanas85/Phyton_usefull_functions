#########################################################################
### get variable name

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

      
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
### Sort Columns By Last Full Year Statistics

   SortedColumnList_DF = list( DF.resample('Y').sum().sort_values(by = DF.resample('Y').sum().index[-2],axis = 1, ascending = False).columns )
   DF = DF[SortedColumnList_DF]


#########################################################################
### from groupby to dicinary

# Slice by delivery date - year contracts 
   DF_ByYearDelivery = dict(tuple(DF.groupby('Year_Delivery')))


#########################################################################
### Join columns into one and drop nan

   DF.apply(lambda x: ','.join(x.dropna()), axis=1)


#########################################################################   
### Function to extract code from GitHub

def GetGitHubCode(GitUrl):

    response = requests.get(GitUrl) #get data from json file located at specified URL 

    if response.status_code == requests.codes.ok:
        contentOfUrl = response.content
        exec(contentOfUrl, globals() )
    else:
        print('Content was not found.')
         
#########################################################################
### File Path depends on system

    if os.name == 'nt': # Windows
        pathWithFileName = os.getcwd() + "\\dataset\\" + fileName
    else:
        pathWithFileName = os.getcwd() + "/dataset/" + fileName
         
         
#########################################################################
### Dictionary to csv with Pandas

   # write
   pd.DataFrame.from_dict(df, orient="index").to_csv(pathWithFileName, header=False)
   
   # read
   pd.read_csv(pathWithFileName, header=None, dtype={0: int, 1:str}).set_index(0).squeeze().to_dict()
   

#########################################################################
### Unique values based on few columns
   
   DF.groupby(by=['col1','col2'], as_index=False).first()


   
      def RemoveLeapYear(DF):
       
       DFi = DF.copy()
       DFi = DFi[~((DFi.index.month == 2) & (DFi.index.day == 29))]
       
       return DFi
   
#########################################################################
### Temperature Month Statistics

   
   def GetTempAndHDDmothStatistics(DF, TemperatureVar ):
       
      # 'DF' dataFrame with datetime index and 'TemperatureVar' temperature variable
       
      # rempove '29-02' date
      DFin = DF.copy()
      DFin = RemoveLeapYear(DFin)
      
      # add variables:

      DFin = DFin.assign(
                          Year = lambda x: x.index.year,
                          Month = lambda x: x.index.month,
                          HDD  = lambda x: np.where( x['Temp_avg'] <= 15,
                                                   ( 18-x['Temp_avg'] ).round(1).astype(float), 0 )
                          )
       
      # define aggregation function:
      def my_agg(x):
            names = {
                'HDD_sum': x['HDD'].sum(),
                'Temp_mean':  x[ TemperatureVar ].mean()}
        
            return pd.Series(names, index=['HDD_sum', 'Temp_mean'])
    
      DFstatistics = DFin.groupby(["Year", "Month"]).apply(my_agg).groupby(["Month"]).agg(['min','mean','max'])

      return DFstatistics


    GetTempAndHDDmothStatistics(TempCityHistory, 'Temp_avg' )



#########################################################################
### Var Name to string

   f'{varName=}'.split('=')[0]
 
