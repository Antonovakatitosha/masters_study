import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn

diseases = pd.read_csv("dermatology.csv")

print(set(diseases[diseases.columns[-1]]))
diseases['disease_class'] = diseases['disease_class'].map({6: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

X = diseases[diseases.columns[:-1]]
Y = diseases[diseases.columns[-1]]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=45)


def build_confusion_matrix(predicted, target):
    matrix = np.zeros((6, 6)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[i]][target[i]] += 1
    print(matrix)
    return matrix


def draw_matrix(matrix):
    plt.figure()
    seaborn.heatmap(matrix, annot=True, fmt='d', cmap="Greens")
    plt.xlabel('real values')
    plt.ylabel('predicted')
    plt.title("Confusion matrix")
    plt.savefig("Confusion matrix")
    # plt.show()


CNB = CategoricalNB()
CNB.fit(x_train, y_train)
prediction_NB = CNB.predict(x_test)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
prediction_DT = decision_tree.predict(x_test)

param = {
    "max_depth": [3, 5, 7],
    "min_samples_leaf" : [3, 4, 5],
    "min_samples_split": [6, 8, 10],
    "class_weight": [{0: 0.67, 1: 0.67, 2: 0.67, 3: 0.67, 4: 0.67, 5: 0.67}, "balanced"]

}
decision_tree_param = GridSearchCV(DecisionTreeClassifier(), param, cv=5, verbose=1)
decision_tree_param.fit(x_train, y_train)
prediction_DTP = decision_tree_param.predict(x_test)
print(decision_tree_param.best_params_)

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
prediction_RF = random_forest.predict(x_test)

confusion_matrix = build_confusion_matrix(prediction_RF, y_test.values)
draw_matrix(confusion_matrix)


## metrics for class 1
def calculate_metrics(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    print(f"\nprecision = {precision}, recall = {recall}, F1_score = {F1_score}")


TP = confusion_matrix[0, 0]
TN = confusion_matrix[1:, 1:].sum()
FP = confusion_matrix[0, 1:].sum()
FN = confusion_matrix[1:, 0].sum()
calculate_metrics(TP, TN, FP, FN)
accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
print(f"accuracy = {accuracy}")





