import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics

qualifies_by_single_grade = pd.read_csv("single_grade.csv")
qualifies_by_single_grade.sort_values(by="grade", inplace=True)
print(qualifies_by_single_grade)

qualified_candidates = qualifies_by_single_grade[qualifies_by_single_grade["qualifies"] == 1]
unqualified_candidates = qualifies_by_single_grade[qualifies_by_single_grade["qualifies"] == 0]

plt.scatter(qualified_candidates["grade"], qualified_candidates["qualifies"], color="g")
plt.scatter(unqualified_candidates["grade"], unqualified_candidates["qualifies"], color="r")

X = np.array(qualifies_by_single_grade["grade"]).reshape(-1, 1)
y = np.array(qualifies_by_single_grade["qualifies"])

print("Logistic model.")

# logistic_model = LogisticRegression()
logistic_model = LogisticRegression(solver="lbfgs")
logistic_model.fit(X, y)
logistic_modeled_values = logistic_model.predict(X)

logistic_prediction_probabilities = logistic_model.predict_proba(X)[:, 1]

qualifies_by_single_grade["logistic probability"] = logistic_prediction_probabilities

logistic_confusion_matrix = metrics.confusion_matrix(y, logistic_modeled_values)
print(logistic_confusion_matrix)

print("Accuracy:", metrics.accuracy_score(y, logistic_modeled_values))
print("Error Rate:", 1 - metrics.accuracy_score(y, logistic_modeled_values))
print("Precision:", metrics.precision_score(y, logistic_modeled_values))
print("Recall:", metrics.recall_score(y, logistic_modeled_values))

print("Linear model.")

linear_model = LinearRegression()
linear_model.fit(X, y)
linear_prediction_probability = linear_model.predict(X)

linear_modeled_values = [int(round(v)) for v in linear_prediction_probability]

qualifies_by_single_grade["linear probability"] = linear_prediction_probability
print(qualifies_by_single_grade)

linear_confusion_matrix = metrics.confusion_matrix(y, linear_modeled_values)
print(linear_confusion_matrix)

print("Accuracy:", metrics.accuracy_score(y, linear_modeled_values))
print("Error Rate:", 1 - metrics.accuracy_score(y, linear_modeled_values))
print("Precision:", metrics.precision_score(y, linear_modeled_values))
print("Recall:", metrics.recall_score(y, linear_modeled_values))

plt.plot(X, linear_modeled_values, color="y")
plt.plot(X, linear_prediction_probability, color="y")
# plt.plot(X, logistic_modeled_values, color="b")
plt.plot(X, logistic_prediction_probabilities, color="b")
plt.show()