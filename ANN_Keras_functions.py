import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
from keras import optimizers
from keras.layers import Dropout
from keras.constraints import maxnorm

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

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


def CreateFeedForwardModel(HiddenLayersNumber=1,\
                           NeuronsNumber=10,\
                           InputShape = (1, ),\
                           AddBatchNorm = False,\
                           LossFun = 'mean_squared_error',\
                           Opt = 'Adam()',\
                           ActivationFun = 'relu',
                           ActivationOut = 'linear',
                           DropoutValue = None,
                           constraintValue = None,
                           init = 'glorot_uniform'):
    
    # create model
    model = Sequential()

    if constraintValue is not None:
        model.add(Dense(NeuronsNumber, input_shape = InputShape,\
                        activation = ActivationFun,\
                        kernel_initializer = init,\
                        kernel_constraint = maxnorm(constraintValue) ) )
    else:
        model.add(Dense(NeuronsNumber, input_shape = InputShape,\
                        activation = ActivationFun,\
                        kernel_initializer = init ) )
        
    if DropoutValue is not None:
        model.add(Dropout( DropoutValue ))

    for ln in range(HiddenLayersNumber-1):
        
        if constraintValue is not None:
            model.add(Dense(NeuronsNumber,\
                            activation=ActivationFun,\
                            kernel_initializer = init,\
                            kernel_constraint = maxnorm(constraintValue) ) )
        else:
            model.add(Dense(NeuronsNumber,\
                            activation=ActivationFun,\
                            kernel_initializer = init ) )

        if AddBatchNorm:
           model.add( BatchNormalization() )
           
        if DropoutValue is not None:
            model.add(Dropout( DropoutValue ))
	
    # add output layer
    model.add(Dense(1, activation = ActivationOut,\
                    kernel_initializer = init ))
    
    Opt = eval(f'tf.optimizers.{Opt}')
    lr_metric = get_lr_metric(Opt)

	# Compile model
    model.compile(loss = LossFun, optimizer=Opt, metrics=[LossFun, lr_metric] )
    
    return model
