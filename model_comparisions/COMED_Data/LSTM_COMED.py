#LSTM RNN Implementation

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

comed = pd.read_csv('COMED_hourly.csv')
comed = comed.sort_values('Datetime')
comed.index = pd.to_datetime(comed['Datetime'], format='%Y-%m-%d %H:%M:%S')
comed = comed.iloc[:,[1]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(comed)
comed = pd.DataFrame(scaled, columns=['COMED'])

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
x,y = df_to_x_y(comed, WINDOW_SIZE)

x = np.reshape(x, (x.shape[0], WINDOW_SIZE, 1))
y = np.reshape(y, (y.shape[0], ))

x_train, y_train = x[:55000],y[:55000]
x_val, y_val = x[55000:60000],y[55000:60000]
x_test, y_test = x[60000:],y[60000:]

from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((10,1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

lr = 0.001
e = 15

cp = ModelCheckpoint('model1/', save_best_only=True)

model1.compile(loss=MeanSquaredError(),optimizer=Adam(learning_rate=lr), metrics=[RootMeanSquaredError()])
history = model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=e, callbacks=[cp])
hist = pd.DataFrame(history.history)

from tensorflow.keras.models import load_model
model1= load_model('model1/')

train_predictions = model1.predict(x_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
plt.plot(train_results['Train Predictions'][:100])
plt.plot(train_results['Actuals'][:100])

val_predictions = model1.predict(x_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
plt.plot(val_results['Val Predictions'][:100], 'green')
plt.plot(val_results['Actuals'][:100], 'blue')

test_predictions = model1.predict(x_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
plt.plot(test_results['Test Predictions'][:100], 'green')
plt.plot(test_results['Actuals'][:100], 'blue')