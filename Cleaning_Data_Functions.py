import pandas as pd

######################################################################
def RemoveTopAndBottomRowsWitAllNA(DFinput):
        
    DF = DFinput.copy()
    for index, row in  DF.iterrows():
        if pd.isnull( row ).all() :
            DF = DF.iloc[1:]
        else :
            break    # break here
        
    # itereate in diffrent order
    for index, row in  DF[::-1].iterrows():
        if pd.isnull( row ).all() :
            DF = DF.iloc[:-1]
        else :
            break    # break here
  
    return DF


######################################################################

def RemoveTopAndBottomRowsWithNA(DF, ColName):
    
    if(DF.index.name == ColName):
        DF.reset_index() 
    
    for index, row in  DF.iterrows():
        if pd.isnull( row[ColName] ) :
            DF = DF.iloc[1:]
        else :
            break    # break here
    
    # itereate in diffrent order
    for index, row in  DF[::-1].iterrows():
        if pd.isnull( row[ColName] ) :
            DF = DF.iloc[:-1]
        else :
            break    # break here

    if(DF.index.name == ColName):
        DF.set_index(ColName)
        
    return(DF)

######################################################################

def RemoveBottomRowsWithNA(DF, ColName):
    
    if(DF.index.name == ColName):
        DF.reset_index() 
    
               
    for index, row in  DF[::-1].iterrows():
        if pd.isnull( row[ColName] ) :
            DF = DF.iloc[:-1]
        else :
            break    # break here

    if(DF.index.name == ColName):
        DF.set_index(ColName)
        
    return(DF)

######################################################################

def RemoveTopRowsWithNA(DF, ColName):
    
    if(DF.index.name == ColName):
        DF.reset_index() 
    
               
    for index, row in  DF.iterrows():
        if pd.isnull( row[ColName] ) :
            DF = DF.iloc[1:]
        else :
            break    # break here

    if(DF.index.name == ColName):
        DF.set_index(ColName)
        
    return(DF)

######################################################################

def CheckDupicatesOnIndex(DF, IndexColumnName):
    if DF.index.is_unique :
        print('DateFrame does not have duplicated dates')  
    else :
        DF = DF.reset_index(). \
            drop_duplicates(subset= IndexColumnName, keep='first') \
            .set_index(IndexColumnName).sort_index()
        print('Duplicated dates in DateFrame has been removed') 
        
    return(DF)
