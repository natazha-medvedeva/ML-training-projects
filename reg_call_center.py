import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
# from mlinsights.mlmodel import QuantileLinearRegression

plt.figure(figsize=(20, 8))
call_center_data = pd.read_csv('call_center.csv', parse_dates=['timestamp'])
# print(call_center_data)
# print(call_center_data.dtypes)

X = np.array([t.value for t in call_center_data['timestamp']]).reshape(-1, 1)
y = np.array(call_center_data['calls']).reshape(-1, 1)

plt.plot(X, y)
# plt.show()

ols_model = lm.LinearRegression()
ols_model.fit(X, y)
ols_trend = ols_model.predict(X)
print(ols_model.coef_)
print(ols_model.intercept_)
print(ols_trend[-1] - ols_trend[0])

plt.plot(X, ols_trend)
plt.show()

