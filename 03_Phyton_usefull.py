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

