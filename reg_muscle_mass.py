import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn

muscle_mass_from_training = pd.read_csv('muscle_mass.csv')
muscle_mass_from_training.sort_values(by='training_time', inplace=True)
# print(muscle_mass_from_training)

X = np.array(muscle_mass_from_training['training_time']).reshape(-1, 1)
y = np.array(muscle_mass_from_training['muscle_mass']).reshape(-1, 1)

plt.scatter(X, y)
# plt.show()

regression_model = lm.LinearRegression()
polynomial_transformer = pp.PolynomialFeatures(degree=2)
X_transformed = polynomial_transformer.fit_transform(X)

# print(X_transformed)

regression_model.fit(X_transformed, y)
modeled_muscle_mass = regression_model.predict(X_transformed)
print(regression_model.coef_)
print(regression_model.intercept_)

plt.plot(X, modeled_muscle_mass, color = 'g')
plt.show()

print(muscle_mass_from_training.columns)