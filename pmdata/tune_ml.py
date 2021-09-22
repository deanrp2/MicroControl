"""
This should be a temporary file!!!
"""
# Basic libs
import pandas as pd
import utils
import numpy as np
from matplotlib import pyplot as plt

# Main ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

# Processing Functions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
seed=1

from neorl.tune import GRIDTUNE

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
data = pd.read_csv("qpower_210916.csv", index_col = 0)

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
#xscaler = StandardScaler()     #x-scaler object
#yscaler = StandardScaler()     #y-scaler object
Xtrain = transform_features(Xtrain, f='cos')
Xtest = transform_features(Xtest, f='cos')
#Ytrain = yscaler.fit_transform(Ytrain)
#Ytest = yscaler.transform(Ytest)


#*********************************************************
# NN model
#*********************************************************

model = Sequential()
model.add(Dense(400,input_dim = Xtrain.shape[1], activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(Ytrain.shape[1], activation='linear'))
model.compile(loss='mean_absolute_error', optimizer=Adam(1e-3), metrics=['mean_absolute_error'])
model.summary()
#checkpoint = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.9, patience=5, min_lr=0, verbose=1)
history=model.fit(Xtrain, Ytrain, epochs=700, batch_size=64, validation_split = 0.2, verbose=True)

print('---(c)---')
Ynn=model.predict(Xtest)
nn_mae=mean_absolute_error(Ytest,Ynn)
nn_rmse=np.sqrt(mean_squared_error(Ytest,Ynn))
nn_r2=r2_score(Ytest,Ynn)
print('NN Summary:', 'MAE=',nn_mae, 'RMSE=',nn_rmse, 'R2=', nn_r2)

train_loss = history.history['mean_absolute_error']
val_loss = history.history['val_mean_absolute_error']
plt.figure()
plt.plot(train_loss, label = 'Training')
plt.plot(val_loss, label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
