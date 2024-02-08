import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier


def create_confusion_matrix(predicted, target):
    matrix = np.zeros((3, 3)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[i]][target[i]] += 1
    return matrix


def plot_matrix(matrix):
    plt.figure()
    # fmt - "format string”, 'd' - decimal
    seaborn.heatmap(matrix, annot=True, cmap='Greens', fmt='d')
    plt.xlabel('real value')
    plt.ylabel('predicted')
    plt.title('Confusion matrix')
    plt.savefig('Confusion matrix.png')
    plt.show()


def plt_tree(tree):
    plt.figure(figsize=(25, 25))
    plot_tree(tree, filled=True)
    plt.savefig('decision_tree.png')
    plt.show()


data = pd.DataFrame(columns=['weather', 'is_play'])
data['weather'] = ['cold', 'wind', 'normal', 'sun']
data['is_play'] = ['no', 'no', 'yes', 'yes']
# print(data)
# print(data.values)
temp = data.loc[(data['is_play'] == 'yes') & (data['weather'] == 'normal')]
# print(temp)


iris = pd.read_csv('../Naive Bayes/bezdekIris.csv')
iris['class'] = iris['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
# iris[iris.columns[-1]] = iris[iris.columns[-1]].map({'Iris-setosa': 0,
#                                                     'Iris-versicolor': 1,
#                                                     'Iris-virginica': 2})

# print(iris)
X = iris[iris.columns[:-1]]
Y = iris[iris.columns[-1]]
# print(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# print(X_test)

# GNB = GaussianNB()
# GNB.fit(X_train, Y_train)
# prediction = GNB.predict(X_test)
# print(prediction, Y_test)
#
# decision_tree = DecisionTreeClassifier()
# decision_tree.fit(X_train, Y_train)
# prediction = decision_tree.predict(X_test)
# print(export_text(decision_tree))
# plt_tree(decision_tree)

# params = {
#     'max_depth': [4, 6, 8, 10],
#     'min_samples_leaf': [2, 5, 7, 10],
#     'min_samples_split':  [2, 5, 7, 10],
#     'class_weight': [{0: 0.33, 1: 0.33, 2: 0.33}, 'balanced'] #  'balanced_subsample'
# }
# decision_tree = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
# decision_tree.fit(X_train, Y_train)
# best_params = decision_tree.best_params_
# print(best_params)
# dcc = decision_tree.best_estimator_ # возвращает DecisionTreeClassifier
# plt_tree(dcc)
# print(dcc)
# prediction = decision_tree.predict(X_test)


params = {'n_estimators': [32, 115, 128, 145],
          'max_depth': [10, 15, 21, 22, 23],
          'min_samples_leaf': [2, 5, 7, 10],
          'min_samples_split': [2, 5, 7, 10],
          'class_weight':  [{0: 0.33, 1: 0.33, 2: 0.33}, 'balanced', 'balanced_subsample']}
random_forest = GridSearchCV(RandomForestClassifier(), params, cv=5)
random_forest.fit(X_train, Y_train)
prediction = random_forest.predict(X_test)


confusion_matrix = create_confusion_matrix(prediction, Y_test.values)
# print(confusion_matrix)
plot_matrix(confusion_matrix)
report = classification_report(prediction, Y_test.values)
print(report)



