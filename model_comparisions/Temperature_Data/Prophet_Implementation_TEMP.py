import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

temp = pd.read_csv('Global_annual_mean_temp.csv',names=['ds', 'y', 'ydash'], header=0)
temp = temp.iloc[:, [0,1]]
temp['ds'] = pd.to_datetime(temp['ds'], format='%Y')
temp.sort_values('ds', inplace=True, ascending=True)
temp['ds'] = temp['ds'].astype(object)

temp_train, temp_test = train_test_split(temp, test_size=0.1, shuffle=False)

m = Prophet()
m.fit(temp_train)

forecast = m.predict(temp_test)

plt.plot(temp_train['ds'], temp_train['y'], 'blue', temp_test['ds'], temp_test['y'], 'orange', forecast['ds'], forecast['yhat'], 'green')
#print(mean_absolute_error(temp_test['y'], forecast['yhat']))
#print(mean_squared_error (temp_test['y'], forecast['yhat']))