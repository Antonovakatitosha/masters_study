import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# function for populating a confusion matrix with values
def form_matrix(y_test, y_pred):
    matrix = np.zeros((2, 2)).astype('int64')
    for i in range(len(y_test)):
        matrix[y_pred[i]][y_test[i]] += 1
    return matrix


# function for displaying the confusion matrix
def display_matrix(matrix):
    plt.figure()
    seaborn.heatmap(matrix, annot=True, cbar=False, cmap='Blues', fmt='d')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title('Confusion matrix')
    plt.show()


# function for manual metric evaluation
def evaluate_metrics(matrix):
    # definition of confusion matrix characteristics
    TP = matrix[0, 0]
    TN = matrix[1, 1]
    FP = matrix[0, 1]
    FN = matrix[1, 0]

    # manual classification metric calculation based on the confusion matrix characteristics
    # accuracy classification is defined in this way only for a matrix of size 2x2
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print('Classification accuracy: ' + str(round(accuracy, 2)))
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


# loading the dataset
data = pd.read_csv('tic-tac-toe.csv')
# print(data['top_left'])

# splitting the dataset into input and output columns
# this is done due to the need to map the values of every column into integer representations
in_cols = data.columns[:-1]
out_cols = data.columns[-1]
# print(data[out_cols])

# mapping of the input column values to integer values
for i in in_cols:
    data[i] = data[i].map({'o': 0, 'x': 1, 'b': 2})

# mapping of the output column values to integer values
data[out_cols] = data[out_cols].map({'negative': 0, 'positive': 1})

# splitting the dataset into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data[in_cols], data[out_cols], test_size=0.2)

# creation of a decision tree classification model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# creation of the confusion matrix
matrix = form_matrix(list(y_test), predicted)

# displaying of the confusion matrix
display_matrix(matrix)

# manual classification metric calculation
evaluate_metrics(matrix)

# automatic, inbuild method for classification metric calculation
from sklearn.metrics import classification_report

print()
report = classification_report(list(y_test), predicted)
print(report)
