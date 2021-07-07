import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


def visualize_single_variable_regression (x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    plt.scatter(x, y)

    linear_model = lm.LinearRegression()
    linear_model.fit(x, y)

    model = linear_model.predict(x)

    plt.plot(x, model, color='r')

    plt.show()

advertising_data = pd.read_csv('advertising.csv', index_col=0)
print(advertising_data)

# visualize_single_variable_regression(advertising_data['TV'], advertising_data['sales'])
# visualize_single_variable_regression(advertising_data['radio'], advertising_data['sales'])
# visualize_single_variable_regression(advertising_data['newspaper'], advertising_data['sales'])

ad_data = np.array(advertising_data[['TV', 'radio', 'newspaper']]).reshape(-1, 3)
sales_data = np.array(advertising_data[['sales']]).reshape(-1, 1)

linear_model = lm.LinearRegression()
linear_model.fit(ad_data, sales_data)

print(linear_model.coef_)
print(linear_model.intercept_)




