import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler


qualifies_by_double_grade = pd.read_csv("double_grade_reevaluated.csv")
print(qualifies_by_double_grade)

qualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 1]
unqualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade["qualifies"] == 0]

plt.xlabel("technical_grade")
plt.ylabel("english_grade")

X = np.array(qualifies_by_double_grade[["technical_grade", "english_grade"]]).reshape(-1, 2)
y = np.array(qualifies_by_double_grade["qualifies"]).reshape(-1, 1)

standart_scaler = StandardScaler()
X = standart_scaler.fit_transform(X)

dummy_encoding = OneHotEncoder(sparse=False, categories="auto")
y = dummy_encoding.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

ann = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000000)
ann.fit(X_train, y_train)

y_predicted = ann.predict(X_test)
y_probabilities = ann.predict_proba(X_test)[:, 1]

max_grade = 101
english_grades_range = list(range(max_grade))
technical_grades_range = list(range(max_grade))
probability_level = np.empty([max_grade, max_grade])
for x in technical_grades_range:
    for y in english_grades_range:
        prediction_point = standart_scaler.transform(np.array([x, y]).reshape(1, -1))
        probability_level[x, y] = ann.predict_proba(prediction_point)[:,1]

plt.contourf(probability_level, cmap="RdYlBu")

plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="g")
plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="r")

y_test_labels = dummy_encoding.inverse_transform(y_test)
y_predicted_labels = dummy_encoding.inverse_transform(y_predicted)

print(confusion_matrix(y_test_labels, y_predicted_labels))
print(classification_report(y_test_labels, y_predicted_labels))

plt.show()
