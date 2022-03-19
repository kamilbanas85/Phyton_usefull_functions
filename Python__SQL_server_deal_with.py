#%% CONNECT TO SQL Server DATABASE
import pyodbc
import sqlalchemy as sa
import urllib


##########################################
##########################################
##########################################
##########################################
### standard - with pyodbc.connect

##############################
### connect to DB

server = 'dashboard-dbserver01.database.windows.net'
database = 'baza01'
username = 'kamilbanas85'
password = 'pasword'
driver= '{ODBC Driver 17 for SQL Server}'


#'trusted-azure' - inisde web-app with identiy:
with pyodbc.connect('Driver='+driver +
                     ';Server='+server + 
                     ";PORT=1443;Database="+ database +
                     ";Authentication=ActiveDirectoryMsi") as conn:

###'trusted':
with pyodbc.connect('Driver='+driver +
                    ';Server='+server + 
                    ";PORT=1443;Database="+ database +
                    ";Trusted_Connection=yes") as conn:
                
### 'password'
with pyodbc.connect('DRIVER='+driver+
                    ';SERVER=tcp:'+server+
                    ';PORT=1433;DATABASE='+database+
                    ';UID='+username+
                    ';PWD='+ password) as conn:

    
##############################
####### inside connection  read data to pandas  DataFrame  
with pyodbc.connect('DRIVER='+driver+
                    ';SERVER=tcp:'+server+
                    ';PORT=1433;DATABASE='+database+
                    ';UID='+username+
                    ';PWD='+ password) as conn:
    
    data = pd.read_sql_query( Sql_query,  conn)


##############################
####### inside connection make query  
with pyodbc.connect('DRIVER='+driver+
                    ';SERVER=tcp:'+server+
                    ';PORT=1433;DATABASE='+database+
                    ';UID='+username+
                    ';PWD='+ password) as conn:

    with conn.cursor() as cursor:
        cursor.execute( Sql_query )
        cursor.commit()
        
######################################################
######################################################
######################################################
### Sql alchemy

### password
params = urllib.parse.quote_plus('DRIVER='+driver+
                                 ';SERVER=tcp:'+server+
                                 ';PORT=1433;DATABASE='+database+
                                 ';UID='+username+
                                 ';PWD='+ password)

### trusted connection 
params = urllib.parse.quote_plus('DRIVER='+driver+
                                 ';SERVER=tcp:'+server+
                                 ';PORT=1433;DATABASE='+database+
                                 ';TRUSTED_CONNECTION=yes')


engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params),
                          fast_executemany=True,\
                          connect_args={'connect_timeout': 10},\
                          echo=False)


###########################
### exectue query

with engine.connect() as con:
    con.execute( sql_query )

###########################
### write dataframe to sql table

df.reset_index()\
    .assign(Date = lambda x: x['Date'].astype('str'))\
    .to_sql(con=engine, schema="dbo", name="tablename",\
            if_exists="append",\
            index=False,\
            chunksize=1000)
        
# append - pevent to delate datatypes, so to load all data frame it requires delate all rows:
        
################################################################
###############################################################   
# to preserve primery key and data types

sql_query1 = '''
DROP TABLE [dbo].[tableNameInSql]

CREATE TABLE tableNameInSql (
    [Date]     VARCHAR(30)  NOT NULL PRIMARY KEY,
    Demand   Decimal(8,1) )
'''
# or
table_name = 'tableNameInSql'

sql_query2 = f'''
TRUNCATE TABLE {table_name}
'''

with engine.connect() as con:
    con.execute(sql_query1)

