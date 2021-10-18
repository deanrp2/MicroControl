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
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Processing Functions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
seed=1
np.random.seed(seed)   #to fix seeding in careful split random.choice!!  

def transform_features(x, f='cos'):
    
    if f=='cos':
        y=np.cos(x)
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

#use tanh transform
#Xtrain = transform_features(Xtrain, f='tanh')
#Xtest = transform_features(Xtest, f='tanh')

#standard scaler
#scaler=StandardScaler()
#Xtrain=scaler.fit_transform(Xtrain)
#Xtest=scaler.transform(Xtest)


#*********************************************************
# Sklearn models
#*********************************************************
 
#Random Forests
rf_model = RandomForestRegressor(max_features = 8, random_state = seed)
rf_model.fit(Xtrain, Ytrain)

#DT model
dt_model = DecisionTreeRegressor(max_features = 8, random_state = seed)
dt_model.fit(Xtrain, Ytrain)

#LR model
lr_model = LinearRegression(normalize=True)
lr_model.fit(Xtrain,Ytrain)
#print(lr_model.score(Xtest,Ytest))

Yrf=rf_model.predict(Xtest)
rf_mae=mean_absolute_error(Ytest,Yrf)
rf_rmse=np.sqrt(mean_squared_error(Ytest,Yrf))
rf_r2=r2_score(Ytest,Yrf)
print('RF Summary:', 'MAE=',rf_mae, 'RMSE=',rf_rmse, 'R2=', rf_r2)

Ydt=dt_model.predict(Xtest)
dt_mae=mean_absolute_error(Ytest,Ydt)
dt_rmse=np.sqrt(mean_squared_error(Ytest,Ydt))
dt_r2=r2_score(Ytest,Ydt)
print('DT Summary:', 'MAE=',dt_mae, 'RMSE=',dt_rmse, 'R2=', dt_r2)

Ylr=lr_model.predict(Xtest)
lr_mae=mean_absolute_error(Ytest,Ylr)
lr_rmse=np.sqrt(mean_squared_error(Ytest,Ylr))
lr_r2=r2_score(Ytest,Ylr)
print('LR Summary:', 'MAE=',lr_mae, 'RMSE=',lr_rmse, 'R2=', lr_r2)


#*********************************************************
# NN model
#*********************************************************
#best network model
model = Sequential()
model.add(Dense(437,input_dim = Xtrain.shape[1], activation='relu'))
model.add(Dense(258, activation='relu'))
model.add(Dense(101, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(35, activation='relu'))
model.add(Dense(Ytrain.shape[1], activation='linear'))
model.compile(loss='mean_absolute_error', optimizer=Adam(9e-4), metrics=['mean_absolute_error'])
model.summary()
mc_cb = ModelCheckpoint(filepath='power_model.h5', monitor='val_mean_absolute_error', mode='min', save_best_only=True, verbose=0)
#lr_cb = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.95, patience=10, min_lr=0, verbose=1)
history=model.fit(Xtrain, Ytrain, epochs=200, batch_size=8, 
                  validation_split = 0.2, callbacks=[mc_cb], verbose=True)

model=load_model('power_model.h5')
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
