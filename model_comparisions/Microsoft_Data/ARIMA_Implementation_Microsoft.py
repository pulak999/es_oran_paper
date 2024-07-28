from pmdarima.arima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split

#stocks = pd.read_csv('Microsoft Stocks.csv', names=['ds','y','y1','y2','y3','y4'], header=0)
stocks = pd.read_csv('Microsoft Stocks.csv', parse_dates=['ds'], names=['ds','y','y1','y2','y3','y4'], header=0)
stocks = stocks.iloc[:,[0,1]]
stocks = stocks.sort_values('ds')
stocks.set_index(stocks['ds'], inplace=True)
stocks = stocks.iloc[:,[1]]

stocks_train, stocks_test = train_test_split(stocks, test_size=0.05, shuffle=False)

model = auto_arima(stocks_train, Seasonal=False, stepwise=False)

predictions = pd.DataFrame(model.predict(n_periods=50), index=stocks_test.index)
predictions.columns = ['yhat']

plt.figure(figsize=(8,5))
plt.plot(stocks_train, label="Training")
plt.plot(stocks_test, label="Test")
plt.plot(predictions, label="Predicted")
plt.legend(loc='upper left')
plt.show()