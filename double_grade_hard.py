import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn import svm
from sklearn import metrics
from joblib import dump

def plot_model(svm_classifier):
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    plotting_step = 50
    xx = np.linspace(xlim[0], xlim[1], plotting_step)
    yy = np.linspace(ylim[0], ylim[1], plotting_step)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm_classifier.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')

qualifies_by_double_grade = pd.read_csv('double_grade_small.csv')
print(qualifies_by_double_grade)

qualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade['qualifies'] == 1]
unqualified_candidates = qualifies_by_double_grade[qualifies_by_double_grade['qualifies'] == 0]

plt.xlabel('technical_grade')
plt.ylabel('english_grade')

plt.scatter(qualified_candidates['technical_grade'], qualified_candidates['english_grade'], color='g')
plt.scatter(unqualified_candidates['technical_grade'], unqualified_candidates['english_grade'], color='r')

X = np.array(qualifies_by_double_grade[['technical_grade', 'english_grade']]).reshape(-1, 2)
y = np.array(qualifies_by_double_grade['qualifies'])

svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X, y)

plot_model(svm_classifier)

plt.show()