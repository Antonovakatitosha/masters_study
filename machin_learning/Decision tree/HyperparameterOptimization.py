import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV


# create the required functions for later use
def evaluate_accuracy(y_true, y_pred):
    counter = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            counter += 1
    accuracy = counter / len(y_true)
    return accuracy


def show_text_representation(tree):
    text_representation = export_text(tree)
    print(text_representation)


def show_graphical_representation(tree):
    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(tree, feature_names=data.columns, filled=True)


# load the dataset
data = pd.read_csv('abalone.csv')

# data contained in the class culumn is categorical-string, it needs to be converted into a numerical notation through integers
# print(data['Sex'])
data['Sex'] = data['Sex'].map({'M': 0, 'F': 1, 'I': 2})
# print(data['Sex'])

# split the loaded dataset into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data[data.columns[1:]], data[data.columns[0]], test_size=0.2)

# deffinition of a decision tree model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
predicted = decision_tree_model.predict(X_test)

# comparison between the obrained output and the wanted ground truth values
accuracy = evaluate_accuracy(list(y_test), predicted)
print(f'The decision tree model achieved a classification accuracy value of: {accuracy}')


# decision tree hyperparameters
# max_depth
# min_samples_leaf
# min_samples_split
# class_weight

# we define multiple values for each decision tree hyperparameter in order to try out multiple combinations
# max_depth parameter defines the maximum depth of the tree in order to reduce the complexity of the structure
max_depth = [4, 6, 8, 10]

# min_samples_leaf parameter defines the minimum number of data samples needed to create a leaf
# if there is not enough samples to create a leaf, then the leaf will be joined with another (bigger) leaf and prune a branch
min_samples_leaf = [2, 5, 7, 10]

# min_samples_split parameter defines the minimum number of data samples needed to create a new branching node
# if there is not enough samples to create a branching condition, then the condition will not be created and a leaf node will be created
min_samples_split = [2, 5, 7, 10]

# class_weights parameter defines the weight of each class
# classes which appear less frequently in the dataset will have higher weights and the model will pay more attention to them during classification
num_classes = len(np.unique(data[data.columns[0]]))

counts = [0, 0, 0]
for i in range(num_classes):
    counts.append(0)

values = list(data[data.columns[0]])
for i in values:
    counts[i] += 1

# we define a state in which the weights are proportional to the number of appearances of that class in the dataset
# Адаптация к Распределению Данных
weights = {0: 0, 1: 0, 2: 0}
for i in range(len(weights)):
    weights[i] = counts[i] / len(data)

# we define a state in which all classes have the same weight
# предполагается, что все классы одинаково важны, независимо от их частоты в данных
no_weights = {0: 0.33, 1: 0.33, 2: 0.33}

class_weights = [no_weights, weights]
# Почему два словаря: Тестирование Различных Весовых Конфигураций. В контексте
# \оптимизации гиперпараметров, например, с использованием GridSearchCV,
# разные наборы весов классов могут быть проверены, чтобы определить,
# какой набор весов обеспечивает лучшую производительность модели.
# В этом случае два словаря представляют две различные стратегии взвешивания классов.

# словарь гиперпараметров, которые нужно оптимизировать
params = {
    # maximum depth of the decision tree. If a tree branches out further in depth,
    # all new branches will be cut off and the final node will be turned
    # into a leaf node when the maximum depth is reached
    'max_depth': max_depth,
    # минимальное количество образцов, необходимых для создания листа дерева
    # If one class has a number of samples smaller than the minimum defined value, all of its samples
    # will be moved to another class which has more samples than the minimum value
    'min_samples_leaf': min_samples_leaf,
    # минимальное количество образцов, необходимых для разделения внутреннего узла
    'min_samples_split': min_samples_split,
    # Для учета несбалансированности классов в наборе данных. Это может привести к тому,
    # что модель будет предвзято относиться к более представленным классам и хуже
    # работать с классами, имеющими меньше примеров.class_weight помогает решить эту проблему,
    # позволяя указать веса для каждого класса. Вот как это работает:
    #
    # Взвешивание Классов: Классам с меньшим количеством образцов присваивается больший вес,
    # а классам с большим количеством образцов - меньший. Это гарантирует, что модель уделяет
    # больше внимания классам с меньшим количеством образцов во время обучения.
    #
    # Влияние на Функцию Потерь: Веса классов влияют на функцию потерь модели.
    # Ошибки на образцах из классов с большим весом оказывают большее влияние на общую
    # функцию потерь, что побуждает модель лучше классифицировать эти меньшие классы.
    'class_weight': class_weights
}

# In order to conduct hyperparameter optimization, we use an optimizer GridSearchCV().
# GridSearchCV requires the definition of a new machine learning model, definition of
# which hyperparameters will be changed and the number of cross validation folds
model = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
# params означает, что он будет искать наилучшие значения гиперпараметров из предоставленного списка
# cv=5 означает, что кросс-валидация будет использовать 5 разделов (или "folds"). Это значит, что на
# каждом шаге тестирования одна пятая данных будет использоваться для тестирования модели,
# а остальные данные — для обучения.

model.fit(X_train, y_train)
predicted = model.predict(X_test)
accuracy = evaluate_accuracy(list(y_test), predicted)
print(f'Classification accuracy after grid search: {accuracy}')


# it is expected that the model will predict more accurately after grid search
# if the reults stay the same, then the default parameters were chosen as the best combination
# if the results get worse than the created combination of hyperparameters do not contain optimal values and need to be changed
params = model.best_params_
print(f'\nbest results were achieved with the following parameters:\n{params}')

dcc = model.best_estimator_
# show_text_representation(dcc)
show_graphical_representation(dcc)
plt.savefig('GridSearchCV.png')
show_graphical_representation(decision_tree_model)
plt.savefig('DecisionTreeClassifier.png')  # Сохранение графика в файл
# plt.show()
