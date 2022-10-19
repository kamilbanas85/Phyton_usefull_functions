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

