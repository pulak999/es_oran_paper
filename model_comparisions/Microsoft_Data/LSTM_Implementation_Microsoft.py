#LSTM RNN Implementation

import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = pd.read_csv('Microsoft_Stocks.csv', parse_dates=['ds'], names=['ds','y','y1','y2','y3','y4'], header=0)
stocks = stocks.iloc[:,[0,1]]
stocks = stocks.sort_values('ds')
stocks.set_index(stocks['ds'], inplace=True)
stocks = stocks.iloc[:,[1]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(stocks)
stocks = pd.DataFrame(scaled, columns=['Microsoft'])

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
x,y = df_to_x_y(stocks, WINDOW_SIZE)

x = np.reshape(x, (x.shape[0], WINDOW_SIZE, 1))
y = np.reshape(y, (y.shape[0], ))

x_train, y_train = x[:7000],y[:7000]
x_val, y_val = x[7000:8000],y[7000:8000]
x_test, y_test = x[8000:],y[8000:]

from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model2 = Sequential()
model2.add(InputLayer((10,1)))
model2.add(LSTM(64))
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))
#model.summary()

cp = ModelCheckpoint('model2/', save_best_only=True)

model2.compile(loss=MeanSquaredError(),optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
model2.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=11, callbacks=[cp])

from tensorflow.keras.models import load_model
model1 = load_model('model2/')

train_predictions = model2.predict(x_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
#train_results.head()
plt.plot(train_results['Train Predictions'][2000:2400], 'green')
plt.plot(train_results['Actuals'][2000:2400], 'blue')

val_predictions = model1.predict(x_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
plt.plot(val_results['Val Predictions'], 'green')
plt.plot(val_results['Actuals'], 'blue')

test_predictions = model1.predict(x_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
plt.plot(test_results['Test Predictions'], 'green')
plt.plot(test_results['Actuals'], 'blue')