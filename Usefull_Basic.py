##################################################################################
### show NA statistics in DataFrame:

# return number of NA per column
   DF.isna().sum() ( DF.isnull().sum() ) 
	
# return percent of NA in column
   DF.isna().mean() * 100 
		
# show rows with NA and without ?
  NAinDF = DF[ DF[ 'Col1' ].isna() ]
	
#  show rows without NA ?
  DFwithNA = DF[ ~DF[ 'Col1' ].isna() ]

##################################################################################
### Besic Exploratory 

   DF.value_counts() #- function to count the number of unique values
   DF.describe(include = ‘all’)  # -  include = ‘all’	- include all columns not only numeric

  .describe() 
  .info()

##################################################################################
# Convert to other type

   DF[‘Col1’] = DF.Col1.astype[‘float', 'int', 'category', 'datetime', 'bool', 'object’]

##################################################################################
# Convert String Column To DateTime:

   DF[‘Col1’] = pd.to_datetime( DF[‘Col1’], errors = ‘coerce’ )

   # errors=’coerce’ – to convert non reconizet data to NA (not produce error)
   # !!! above give sometime error, so:

   DF.loc[:, ‘Col1’] = pd.to_datetime(DF[‘Col1’] )


##################################################################################
# CONVERT numeric column with ‘,’ instead ‘.’ As decimal separater:

   DF = DF.assign( value = lambda x: pd.to_numeric( x[‘value’].astype(str).str.replace(‘,’ , ‘.’), errors =’coerce’)   )
	


##################################################################################
# Replace 'F' in column names with 'C': temps_c.columns

DF.columns = DF.columns.str.replace('F', 'C')

##################################################################################
# FILTER:
	DF.loc[ DF[‘Col’] > 88 , ‘Col’]
DF.query(‘Col > 88’)



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
### Convert 1 dimensional axis objects into scalars. Series or DataFrames with a single element are squeezed to a scalar.
### DataFrames with a single column or a single row are squeezed to a Series. Otherwise the object is unchanged.

squeeze

##################################################################################
# select columns by regular expression

df.filter(regex='e$', axis=1)

# select rows containing 'bbi'

df.filter(like='bbi', axis=0)



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
	.assign( mean = lambda x: x.groupby( x.index.month ).transform('mean'),
		 A    = lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.25) ),
  		 B    = lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.5) )
		})






##################################################################################
### String in few lines

MyString=\
	‘’’ 
	ijijijijijijjij
	Jjjjkkjjjkkjjkkjjk
	‘’’.replace(‘\n’,’’)

##################################################################################	
### ZIP - return list of pair – tuple:

	zip(FirstIteratableObject, SecondIteratableObject) 
	
##################################################################################	
# Wrap some function with const parameter:

from functool import partial
New_function = partial( someFunction, arg1 = 2 )

##################################################################################
# STRING COMPARISION – finid simillar words etc.:

   from fuzzywuzzy import fuzz 

  # Compare strings:
   Fuzz.WRatio(‘Reading’, ‘Reedaing’)



