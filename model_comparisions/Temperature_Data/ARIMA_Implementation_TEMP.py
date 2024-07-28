from pmdarima.arima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split

temp = pd.read_csv('Global_annual_mean_temp.csv')
temp.sort_values('Year', inplace=True, ascending=True)
temp.index = pd.to_datetime(temp['Year'], format='%Y')
temp = temp.iloc[:,[1]]

temp_train, temp_test = train_test_split(temp, test_size=0.1, shuffle=False)

model = auto_arima(temp_train, Seasonal=False, stepwise=False)

predictions = pd.DataFrame(model.predict(n_periods=100), index=temp_test.index)
predictions.columns = ['yhat']

plt.figure(figsize=(8,5))
plt.plot(temp_train, label="Training")
plt.plot(temp_test, label="Test")
plt.plot(predictions, label="Predicted")
plt.legend(loc='upper left')
plt.show()