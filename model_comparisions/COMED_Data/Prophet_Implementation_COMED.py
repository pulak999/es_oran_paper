import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

comed = pd.read_csv('COMED_hourly.csv', parse_dates=['ds'], names=['ds', 'y'], header=0)
comed.sort_values('ds', inplace=True, ascending=True)
comed_train, comed_test = train_test_split(comed, test_size=0.25, shuffle=False)

m = Prophet()
m.fit(comed_train)

forecast = m.predict(comed_test)
#m.plot_components(forecast)

plt.plot(comed_train['ds'], comed_train['y'], 'blue', comed_test['ds'], comed_test['y'], 'orange', forecast['ds'], forecast['yhat'], 'green')

print(mean_absolute_error(comed_test['y'], forecast['yhat']))
print(mean_squared_error (comed_test['y'], forecast['yhat']))