import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization

##########################################################

def PlotLossTrainVsVal(History):
    
    plt.plot(History.history['val_loss'], 'r',  label='test')
    plt.plot(History.history['loss'], 'b',label='train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation score')
    plt.show()

###########################################################

def PlotAccuracyTrainVsVal(History):
    
    plt.plot(History.history['val_accuracy'], 'r',  label='test')
    plt.plot(History.history['accuracy'], 'b',label='train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.show()
    
############################################################

def CreateFeedForwardModel(HiddenLayersNumber=1,\
                           NeuronsNumber=10,\
                           InputShape = (1, ),\
                           AddBatchNorm = False,\
                           LossFun = 'mean_squared_error',\
                           Opt = 'adam',\
                           ActivationFun = 'relu',
                           ActivationOut = 'linear'):
    
    
	# create model
    model = Sequential()
    model.add(Dense(NeuronsNumber, input_shape = InputShape,\
                    activation=ActivationFun ) )

    for ln in range(HiddenLayersNumber-1):
        
        model.add(Dense(NeuronsNumber, activation=ActivationFun) )
        if AddBatchNorm:
           model.add( BatchNormalization() ) 
	
    # add output layer
    model.add(Dense(1, activation = ActivationOut))
    
	# Compile model
    model.compile(loss = LossFun, optimizer=Opt, metrics=[LossFun] )
    
    return model
