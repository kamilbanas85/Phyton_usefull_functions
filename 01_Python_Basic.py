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

####################################################
## nan find

def nans(df):
   return df[df.isnull().any(axis=1)]

####################################################
## Duplicates


def ShowDupicates(df):

	return df[df.duplicated(keep=False)]	
	
##################################################################################
### Besic Exploratory 

   DF.value_counts() #- function to count the number of unique values
   DF.describe(include = ‘all’)  # -  include = ‘all’	- include all columns not only numeric

  .describe() 
  .info()

##################################################################################
# Convert to other types: ‘float', 'int', 'category', 'datetime', 'bool', 'object’

   DF[‘Col1’] = DF.Col1.astype[‘float']	       
			       
##################################################################################
# Convert String Column To DateTime:

   DF[‘Col1’] = pd.to_datetime( DF[‘Col1’], errors = ‘coerce’ )

   # errors=’coerce’ – to convert non reconizet data to NA (not produce error)
   # !!! above give sometime error, so:

   DF.loc[:, ‘Col1’] = pd.to_datetime(DF[‘Col1’] )
   # or
   DF.loc[:,‘Col1’]  = pd.to_datetime( DF.loc[:,‘Col1’], utc=False )
			    

			       		       
   ### In line
   DF.assign(Date = lambda x: pd.to_datetime( x['Date'], utc=False ) )
   DF.assign(Date = lambda x: pd.to_datetime( x['Date'] ).dt.date )
   DF.assign(Date = lambda x: pd.to_datetime( x['Date'] ).dt.normalize() )			       


#########################################################################
### Extract date from string:

import dateutil.parser as dparser

dparser.parse( 'EEX Auction Calendar_21.12.2020.xlsx', fuzzy=True)
# Output:   datetime.datetime(2020, 12, 21, 0, 0)
			       
##################################################################################
### check if string is date

from dateutil.parser import parse

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
			       

##################################################################################
# CONVERT numeric column with ‘,’ instead ‘.’ As decimal separater:

   DF = DF.assign( value = lambda x: pd.to_numeric( x[‘value’].astype(str).str.replace(‘,’ , ‘.’), errors =’coerce’)   )
	
##################################################################################################
### replace some sign with other

   .replace(' ','', regex=True)  

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

DF = Df\
    	.assign( **{ 'mean' : np.nan, '25%' : np.nan, '50%': np.nan, '75%': np.nan,
            	     'mean+std' : np.nan, 'mean-std' : np.nan} )\
    	.assign( **{
        	     'mean' : lambda x: x.groupby(x.index.month).transform('mean'),
       		     '25%' : lambda x: x.groupby(x.index.month).transform(lambda y: y.quantile(0.25) ),
       		     '50%' : lambda x: x.groupby(x.index.month).transform(lambda y: y.quantile(0.50) ),
       		     '75%' : lambda x: x.groupby(x.index.month).transform(lambda y: y.quantile(0.75) ),
       		     'mean+std' : lambda x: x.groupby(x.index.month).transform(lambda y: y.mean() + y.std() ),
      		     'mean-std' : lambda x: x.groupby(x.index.month).transform(lambda y: y.mean() - y.std() )\
                   } )			       
			       
			       
# with normal names

DF = DF\
	.assign( mean = np.nan, A = np.nan, B = np.nan } )\
	.assign( mean = lambda x: x.groupby( x.index.month ).transform('mean'),
		 A    = lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.25) ),
  		 B    = lambda x: x.groupby( x.index.month ).transform( lambda y: y.quantile(0.5) )
		})

			       
# assign variable with variable name

DF = DF.assign(  **{f'WTI_{AveragePeriod}': lambda x: x.resample('M').mean()['WTI'].rolling( AveragePeriod ).mean()}  )



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


#########################################################################
### Return whether the string can be interpreted as a date

from dateutil.parser import parse

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
			       
#########################################################################
### Extract date from string:

import dateutil.parser as dparser

dparser.parse( 'EEX Auction Calendar_21.12.2020.xlsx', fuzzy=True)
# Output:   datetime.datetime(2020, 12, 21, 0, 0)	       

			       
#########################################################################
### Columns Name to row
			       
   def AddColumnsNameAsRow(DF):
    
      DFtoRetrun = DF.copy()
      DFtoRetrun = pd.concat([pd.DataFrame([DF.columns.to_list()], columns = DF.columns), DF])
      return DFtoRetrun			       

#########################################################################
### Set column names as 1 row

   DF = DF.T.reset_index().T			       

##################################################################################################
### remove multi-index name

.pivot_table(index=['dateCET'], columns = 'scenario', values='value').rename_axis(None, axis=1)			       
			       
#########################################################################
## Iterate over rows:

# row[1] represent data rows
# row[0] represent index of row

for row in Df.iterrows():
     if 'Week' in str(row[1][0]):

#########################################################################			       
#### Filter by data dosn't contain word

Data1[~Data1['Auction_Name'].str.contains("German")]			 
