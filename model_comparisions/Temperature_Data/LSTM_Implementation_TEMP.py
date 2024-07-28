#LSTM RNN Implementation

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

temp = pd.read_csv('Global_annual_mean_temp.csv')
temp.sort_values('Year', inplace=True, ascending=True)
temp.index = pd.to_datetime(temp['Year'], format='%Y')
temp = temp.iloc[:,[1]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(temp)
temp = pd.DataFrame(scaled, columns=['Temperature'])

#x, y
#[[[1],[2],[3],[4],[5]]], [6]
#[[[2],[3],[4],[5],[6]]], [7]
#[[[3],[4],[5],[6],[7]]], [8]

#window_size --> amount of initial observations used for training

def df_to_x_y(df,window_size):
    df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        x.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(x),np.array(y)

WINDOW_SIZE = 10
x,y = df_to_x_y(temp, WINDOW_SIZE)

x = np.reshape(x, (x.shape[0], WINDOW_SIZE, 1))
y = np.reshape(y, (y.shape[0], ))

x_train, y_train = x[:105],y[:105]
x_val, y_val = x[110:115],y[110:115]
x_test, y_test = x[115:],y[115:]

from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model3 = Sequential()
model3.add(InputLayer((10,1)))
model3.add(LSTM(64))
model3.add(Dense(8, 'relu'))
model3.add(Dense(1, 'linear'))

lr = 0.005
e = 80

cp = ModelCheckpoint('model3/', save_best_only=True)

model3.compile(loss=MeanSquaredError(),optimizer=Adam(learning_rate=lr), metrics=[RootMeanSquaredError()])
history = model3.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=e, callbacks=[cp])
hist = pd.DataFrame(history.history)

from tensorflow.keras.models import load_model
model1= load_model('model3/')

train_predictions = model1.predict(x_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
plt.plot(train_results['Train Predictions'], 'green')
plt.plot(train_results['Actuals'], 'blue')

val_predictions = model1.predict(x_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
plt.plot(val_results['Val Predictions'])
plt.plot(val_results['Actuals'])

test_predictions = model1.predict(x_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
plt.plot(test_results['Test Predictions'])
plt.plot(test_results['Actuals'])