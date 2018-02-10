# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:33:15 2018
@author: Rob

"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#load data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter = ',')
X = dataset[:,0:8]
Y = dataset[:,8]

#define model
model = Sequential()
model.add(Dense(12, input_dim = 8,
                activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#compile model
model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

#hard work
model.fit(X,Y,epochs = 150, batch_size = 10)