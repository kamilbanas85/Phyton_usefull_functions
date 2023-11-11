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
from keras.constraints import MaxNorm

##########################################################
###########################################################
###

def PlotLossTrainVsVal(History):
    
    plt.plot(History.history['val_loss'], 'r',  label='test')
    plt.plot(History.history['loss'], 'b',label='train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation score')
    plt.show()

###########################################################
###########################################################
### 

def PlotAccuracyTrainVsVal(History):
    
    plt.plot(History.history['val_accuracy'], 'r',  label='test')
    plt.plot(History.history['accuracy'], 'b',label='train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy score')
    plt.show()

###########################################################
###########################################################
### 

def create_feed_forward_model(hidden_layers_nr = 1,
                              neurons_nr = 10,
                              loss = 'mean_squared_error',
                              optimizer = 'adam',
                              input_shape = (1, ),
                              output_nodes_nr = 1,
                              add_batch_norm = False,
                              activation_fun = 'relu',
                              activation_out = 'linear',
                              dropout = None,
                              constraint_value = None,
                              init = 'glorot_uniform',
                              regression_type = True):

    ##############################################
    # init model
    model = Sequential()
    
    ##############################################
    # 1 layer - imput layer
    if constraint_value is not None:
        model.add(Dense(neurons_nr, input_shape = input_shape,\
                        activation = activation_fun,\
                        kernel_initializer = init,\
                        kernel_constraint = MaxNorm(constraint_value) ) )
    else:
        model.add(Dense(neurons_nr, input_shape = input_shape,\
                        activation = activation_fun,\
                        kernel_initializer = init ) )
        
    if dropout is not None:
        model.add(Dropout( dropout ))
    
    
    ##############################################
    # mid layers
    for ln in range(hidden_layers_nr-1):
        
        if constraint_value is not None:
            model.add(Dense(neurons_nr,\
                            activation = activation_fun,\
                            kernel_initializer = init,\
                            kernel_constraint = MaxNorm(constraint_value) ) )
        else:
            model.add(Dense(neurons_nr,\
                            activation = activation_fun,\
                            kernel_initializer = init ) )

        if add_batch_norm:
           model.add( BatchNormalization() )
           
        if dropout is not None:
            model.add(Dropout( dropout ))
	
    
    ##############################################
    # output layer
    model.add(Dense(output_nodes_nr, activation = activation_out,\
                    kernel_initializer = init ))
    
    #Opt = eval(f'tf.optimizers.{Opt}')
    #lr_metric = get_lr_metric(Opt)


    ##############################################
	# Compile model:
    if regression_type == True:
        model.compile(loss = loss, optimizer = optimizer, metrics=['mse', 'mae', 'mape'] )
    else:
        model.compile(loss = loss, optimizer = optimizer, metrics=["accuracy", "AUC"] )



    return model





############################################################
###########################################################
### old-version

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
                        kernel_constraint = MaxNorm(constraintValue) ) )
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
                            kernel_constraint = MaxNorm(constraintValue) ) )
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




