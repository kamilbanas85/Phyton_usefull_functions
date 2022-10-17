##################################################################################

##################################################################################
### Convert columns without one

for col in DF.loc[ :, DF.columns != 'ColName' ].columns:
      DF.loc[:,col] = pd.to_numeric( DF.loc[:,col], errors='coerce' )

# or 

listToConvert = [x for x in DF.columns.to_list() if x not in ('Col1Name','Col2Name') ]

for col in listToConvert:
    DF.loc[:,col] = pd.to_numeric( DF.loc[:,col], errors='coerce' )
    
##################################################################################
## flatten nested list
 
sum(listName,[])

##################################################################################
### Assign working day

from workalender.europe import Germany
callWorkDayData = Germany()

DF = DF.\
		  assign(WorkDay = lambda x: x.index.to_series().transform(lambda y: callWorkDayData.is_working_day(y)).astype(int) ) )

##################################################################################
### Assign Week Number Or month

DF = DF\
       .assign(week = lambda x: x.index.isocalender().week,
               month = lambda x: x.index.month)


##################################################################################
### Assign month statistics:

# With difficult names

DF = DF\
	.assign( **{ 'mean':np.nan', '25%': np.nan, '50%': np.nan } )\
	.assign( **{
		      'mean': lambda x: x.groupby( x.index.month ).transform('mean'),
		      '25%' : lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.25) ),
  		      '50%' : lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.5) )
		})

# with normal names

DF = DF\
	.assign( mean = np.nan, A = np.nan, B = np.nan } )\
	.assign( mean'= lambda x: x.groupby( x.index.month ).transform('mean'),
		 A    = lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.25) ),
  		 B    = lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.5) )
		})
