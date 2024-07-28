import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

from prophet import Prophet

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

stocks = pd.read_csv('Microsoft_Stocks.csv', parse_dates=['ds'], names=['ds','y','y1','y2','y3','y4'], header=0)
stocks = stocks.iloc[:,[0,1]]
stocks = stocks.sort_values('ds')
stocks['ds'] = stocks['ds'].astype(object)

#stocks.set_index(stocks['ds'], inplace=True)
#stocks = stocks.iloc[:,[1]]

stocks_train, stocks_test = train_test_split(stocks, test_size=0.1, shuffle=False)

m = Prophet()
m.fit(stocks_train)

forecast = m.predict(stocks_test)
#m.plot_components(forecast)

plt.plot(stocks_train['ds'][5000:], stocks_train['y'][5000:], 'blue', stocks_test['ds'], stocks_test['y'], 'orange', forecast['ds'], forecast['yhat'], 'green')

print(mean_absolute_error(stocks_test['y'], forecast['yhat']))
print(mean_squared_error (stocks_test['y'], forecast['yhat']))