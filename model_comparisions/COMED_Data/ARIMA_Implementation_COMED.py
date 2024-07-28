from pmdarima.arima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

from sklearn.model_selection import train_test_split

comed = pd.read_csv('COMED_hourly.csv', parse_dates=['ds'], names=['ds', 'y'], header=0)
comed.sort_values('ds', inplace=True, ascending=True)
comed.set_index(comed['ds'], inplace=True)
comed = comed.iloc[:,[1]]

comed_train, comed_test = train_test_split(comed, test_size=0.05, shuffle=False)

model = auto_arima(comed_train, Seasonal=False, stepwise=False)

predictions = pd.DataFrame(model.predict(n_periods=50), index=comed_test.index)
predictions.columns = ['yhat']

plt.figure(figsize=(8,5))
plt.plot(comed_train, label="Training")
plt.plot(comed_test, label="Test")
plt.plot(predictions, label="Predicted")
plt.legend(loc='upper left')
plt.show()