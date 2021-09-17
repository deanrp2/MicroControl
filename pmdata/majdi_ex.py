"""
This should be a temporary file!!!
"""

import pandas as pd
import utils
import numpy as np
from sklearn.linear_model import LinearRegression

#load data
data = pd.read_csv("qpower_210916.csv", index_col = 0)

#clean data
data = utils.qpower_preprocess(data)

#train and test split
train, test = utils.careful_split(data, tstfrac = 0.3)

#define predictors and responses
pred = ["theta" + str(i) for i in range(1, 9)]
resp = "fluxQ1"

#naive linear model
model = LinearRegression().fit(train[pred], train[resp])
print("Naive Linear Model:")
print("    Training R2", model.score(train[pred], train[resp]))
print("    Testing R2", model.score(test[pred], test[resp]))

#sneaky coordinate transformation
model = LinearRegression().fit(np.cos(train[pred]), train[resp])
print("Sneaky Linear Model:")
print("    Training R2", model.score(np.cos(train[pred]), train[resp]))
print("    Testing R2", model.score(np.cos(test[pred]), test[resp]))
