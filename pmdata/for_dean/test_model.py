# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:59:05 2021

@author: majdi
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def transform_features(x, f='cos'):
    
    if f=='cos':
        y=np.cos(x)
    elif f=='sin':
        y=np.sin(x)
    elif f=='tanh':
        y=np.tanh(x)
    
    return y


model=load_model('power_model.h5')

#make a prediction for a drum angle sample from the dataset
x=np.array([ 2.23750729, -1.69816588,  2.43345342,  1.25377193,  3.14039907,
       -3.08282896,  2.75485086,  0.38679979])

#transform data by cosine transform
xn=transform_features(x)

#predict the power
y=model.predict(np.array([xn, ])).flatten()  #this np/flatten trick to force keras making a prediction of one sample
print('quad flux:', y)

#make power curves
thetas = np.linspace(0, 2*np.pi, 200)
qpowers = np.zeros((thetas.size, 4))
for i in range(thetas.size):
    t = np.zeros(8)
    t[0] += thetas[i]
    t[1] += thetas[i]
    t = transform_features(t)
    qpowers[i] = model.predict(np.array([t, ])).flatten()

for i in range(4):
    plt.plot(thetas, qpowers[:, i], label = "Q" + str(i + 1))
plt.legend()
plt.show()
