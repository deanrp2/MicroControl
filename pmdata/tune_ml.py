# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:17:16 2021

@author: Majdi
"""

# Basic libs
import pandas as pd
import utils
import numpy as np
from matplotlib import pyplot as plt

# Main ML models
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Processing Functions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
seed=1

from neorl.tune import GRIDTUNE, RANDTUNE

tune_method='grid'

def transform_features(x, f='cos'):
    
    if f=='cos':
        y=np.cos(x)
    elif f=='sin':
        y=np.sin(x)
    elif f=='tanh':
        y=np.tanh(x)
    
    return y

#*********************************************************
# Data prep
#*********************************************************

#load data
data = pd.read_csv("qpower_full.csv", index_col = 0)

#clean data
data = utils.qpower_preprocess(data)

#train and test split
train, test = utils.careful_split(data, tstfrac = 0.2)

print(train.shape)
print(test.shape)
print(train.columns)

#define predictors and responses
pred = ["theta" + str(i) for i in range(1, 9)]
resp = ["fluxQ"+str(i) for i in range(1,5)]

#split into X/Y data arrays
Xtrain=train[pred].values
Xtest=test[pred].values

Ytrain=train[resp].values
Ytest=test[resp].values

#input/output scaling 
#Xtrain = transform_features(Xtrain, f='tanh')
#Xtest = transform_features(Xtest, f='tanh')

#scaler=StandardScaler()
#Xtrain=scaler.fit_transform(Xtrain)
#Xtest=scaler.transform(Xtest)

#*********************************************************
# NN model
#*********************************************************
def train_model(num_layers, layer1, layer2, layer3, layer4, layer5, lr, batch_size):
    caseid=np.random.randint(1,10000)
    model_path='best_model_{}.h5'.format(caseid)
    
    if num_layers < 3:
        layer3=0
        layer4=0
        layer5=0
        
    if num_layers < 4:
        layer4=0
        layer5=0
    
    if num_layers < 5:
        layer5=0
          
    model = Sequential()
    model.add(Dense(layer1,input_dim = Xtrain.shape[1], activation='relu'))
    model.add(Dense(layer2, activation='relu'))

    if layer3 != 0:
        model.add(Dense(layer3, activation='relu'))
        
    if layer4 != 0:
        model.add(Dense(layer4, activation='relu'))
    
    if layer5 != 0:
        model.add(Dense(layer5, activation='relu'))
        
    model.add(Dense(Ytrain.shape[1], activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr), metrics=['mean_absolute_error'])
    #model.summary()
    #lr_cb = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.9, patience=5, min_lr=0, verbose=1)
    mc_cb = ModelCheckpoint(filepath=model_path, monitor='val_mean_absolute_error', mode='min', save_best_only=True, verbose=0)
    history=model.fit(Xtrain, Ytrain, epochs=200, batch_size=batch_size, 
                      callbacks=[mc_cb],
                      validation_split = 0.2, verbose=False)
    
    model=load_model(model_path)

    Ynn=model.predict(Xtest)
    mae=mean_absolute_error(Ytest,Ynn)
    rmse=np.sqrt(mean_squared_error(Ytest,Ynn))
    R2=r2_score(Ytest,Ynn)
    print('Parameters:', (num_layers, layer1, layer2, layer3, layer4, layer5, lr, batch_size), 'R2=', R2)
    
    os.remove(model_path)
    
    return R2

if __name__ == '__main__':

    #*************************************************************
    # Step 1 (Tune Layers/Nodes) with Random Search
    #*************************************************************
   
    if tune_method=='random': 
        param_grid={
        #order: (num_layers, layer1, layer2, layer3, layer4, layer5, lr, batch_size):
        'num_layers': ['int', 2, 5],
        'layer1': ['int', 300, 500],  
        'layer2': ['int', 200, 300],  
        'layer3': ['int', 100, 200],
        'layer4': ['int', 50, 100],
        'layer5': ['int', 25, 50],
        'lr': ['float', 9e-4, 1e-3],  #keep this almost fixed and tune them with grid later
        'batch_size': ['grid', (64, 64)]} #keep this almost fixed and tune them with grid later
        
        #setup a random tune object
        rtune=RANDTUNE(param_grid=param_grid, fit=train_model, ncases=200, seed=1)
        #view the generated cases before running them
        print(rtune.hyperparameter_cases)
        #tune the parameters with method .tune
        randres=rtune.tune(ncores=32, csvname='rand_tune.csv')
        randres = randres.sort_values(['score'], axis='index', ascending=False)
        print(randres)
         
    #*************************************************************
    # Step 2 Refine with grid search
    #*************************************************************
        
    if tune_method=='grid': 
        param_grid={
        #order: (num_layers, layer1, layer2, layer3, layer4, layer5, lr, batch_size):
        'num_layers': [5],
        'layer1': [437],  
        'layer2': [258],  
        'layer3': [101],
        'layer4': [75],
        'layer5': [35],
        'lr': [1e-3, 9e-4, 8e-4, 7e-4],
        'batch_size': [8, 16, 32, 64]}  
        
        #setup a grid tune object
        gtune=GRIDTUNE(param_grid=param_grid, fit=train_model)
        #view the generated cases before running them
        print(gtune.hyperparameter_cases)
        #tune the parameters with method .tune
        gridres=gtune.tune(ncores=32, csvname='grid_tune.csv')
        gridres = gridres.sort_values(['score'], axis='index', ascending=False)
        print(gridres)

