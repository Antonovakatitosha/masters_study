import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def form_matrix(y_test, y_pred):
    matrix = np.zeros((3, 3))
    for i in range(len(y_test)):
        matrix[y_pred[i]][y_test[i]] += 1
    return matrix.astype('uint64')


def display_matrix(matrix):
    plt.figure()
    seaborn.heatmap(matrix, annot=True, cbar=False, cmap='Blues', fmt='d')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title('Confusion matrix')
    plt.show()


def evaluate_metrics(matrix):
    # the accuracy metric is not tied to any specific class, and is instead a gloabl metric
    # it is calculated by dividing the sum of the main diagonal with the sum of all elements of the matrix
    diagonal = 0

    for i in range(len(matrix)):
        diagonal += matrix[i, i]
    accuracy = diagonal / matrix.sum()
    print('Achieved classification accuracy: ' + str(round(accuracy, 2)))
    print()

    # class 0

    # true positive for class 0 is an indicator of how many times the model correcly predicted class 0 when 0 was the expected outcome
    TP = matrix[0, 0]
    # true negative for class 0 is an indicator of how many times the model predicted any class other than 0 when the expected outcome was any class other than 0
    # when calculating metrics for class 0 the truth values of other classes are not taken into account
    TN = matrix[1:, 1:].sum()
    # false positive (first degree error) is an indicator of how many times the model predicted class 0 when the expected outcome was any other class
    FP = matrix[0, 1:].sum()
    # false negative (second degree error) is an indicator of how many times any other class was predicted when the expected outcome was class 0
    FN = matrix[1:, 0].sum()

    # the calculation of metrics is dependent on the chosen values of TP, TN, FP and FN and does not depend on the class after the choice is made
    print('Class 0 metrics: ')
    precision = TP / (TP + FP)
    print('Precision: ' + str(round(precision, 2)))
    NPV = TN / (TN + FN)
    print('Negative predictive value: ' + str(round(NPV, 2)))
    recall = TP / (TP + FN)
    print('Recall: ' + str(round(recall, 2)))
    specificity = TN / (TN + FP)
    print('Speicificty: ' + str(round(specificity, 2)))
    f1 = (2 * precision * recall) / (precision + recall)
    print('f1 score: ' + str(round(f1, 2)))
    print()

    # class 1

    TP = matrix[1, 1]
    TN = matrix[0, 0] + matrix[0, 2] + matrix[2, 0] + matrix[2, 2]
    FP = matrix[1, 0] + matrix[1, 2]
    FN = matrix[0, 1] + matrix[2, 1]

    print('Class 1 metrics: ')
    precision = TP / (TP + FP)
    print('Precision: ' + str(round(precision, 2)))
    NPV = TN / (TN + FN)
    print('Negative predictive value: ' + str(round(NPV, 2)))
    recall = TP / (TP + FN)
    print('Recall: ' + str(round(recall, 2)))
    specificity = TN / (TN + FP)
    print('Speicificty: ' + str(round(specificity, 2)))
    f1 = (2 * precision * recall) / (precision + recall)
    print('f1 score: ' + str(round(f1, 2)))
    print()

    # class 2

    TP = matrix[2, 2]
    TN = matrix[:2, :2].sum()
    FP = matrix[2, :2].sum()
    FN = matrix[:2, 2].sum()

    print('Class 2 metrics: ')
    precision = TP / (TP + FP)
    print('Precision: ' + str(round(precision, 2)))
    NPV = TN / (TN + FN)
    print('Negative predictive value: ' + str(round(NPV, 2)))
    recall = TP / (TP + FN)
    print('Recall: ' + str(round(recall, 2)))
    specificity = TN / (TN + FP)
    print('Speicificty: ' + str(round(specificity, 2)))
    f1 = (2 * precision * recall) / (precision + recall)
    print('f1 score: ' + str(round(f1, 2)))
    print()


data = pd.read_csv('bezdekIris.csv')

data[data.columns[-1]] = data[data.columns[-1]].map({'Iris-setosa': 0,
                                                     'Iris-versicolor': 1,
                                                     'Iris-virginica': 2})

X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]], test_size=0.5)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

matrix = form_matrix(list(y_test), predicted)

display_matrix(matrix)

evaluate_metrics(matrix)

from sklearn.metrics import classification_report

print()
report = classification_report(list(y_test), predicted)
print(report)
