import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# create the required functions for later use
def evaluate_accuracy(y_true, y_pred):
    counter = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            counter += 1
    accuracy = counter / len(y_true)
    return accuracy


# load the dataset
data = pd.read_csv('bezdekIris.csv')


# data contained in the class column is categorical-string,
# it needs to be converted into a numerical notation through integers
# print(data['class'])
data['class'] = data['class'].map({'Iris-setosa': 0,
                                   'Iris-versicolor': 1,
                                   'Iris-virginica': 2})
# print(data['class'])

# split the loaded dataset into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]], test_size=0.2)

# deffinition of a decision tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
predicted = decision_tree_model.predict(X_test)

# comparison between the obrained output and the wanted ground truth values
accuracy = evaluate_accuracy(list(y_test), predicted)
print()
print('The decision tree model achieved a classification accuracy value of: ')
print(accuracy)
print()


# text representation of the created decision tree model
# the text representation is printed in the console
text_representation = export_text(decision_tree_model)
print(text_representation)

# graphical representation of the created decision tree model
# the graphical representation is printed where all other plots are drawn (depends on the version of spyder that you are using)

fig = plt.figure(figsize=(25, 25))
_ = plot_tree(decision_tree_model, feature_names=data.columns, filled=True)
fig.savefig('decision_tree.png')  # Сохранение графика в файл
# plt.show()
