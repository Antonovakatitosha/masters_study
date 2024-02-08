import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def evaluate_accuracy(y_true, y_pred):
    counter = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            counter += 1
    accuracy = counter / len(y_pred)
    return accuracy


data = pd.read_csv('winequality_red.csv')

X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]], test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

predicted = rf.predict(X_test)
accuracy = evaluate_accuracy(list(y_test), predicted)
print(f'\nRandom Forest classifier achieved a classification accuracy value of: {accuracy}\n')

###################################################################################################################

# load the winequality_white dataset
# split the set into training and test subsets
# create a random forest clasisifier model with default parameters
# train the model using training data
# test the model using test data
# evaluate the results

###################################################################################################################
# random forest model parameters

# n_estimators parameter defines the number of decision trees which will be created
# Larger number of trees allows for more subsets of the original dataset to be created,
# but it also extends the time needed for training and decision-making.
n_estimators = [32, 115, 128, 145]

# max_depth parameter defines the maximum depth of all decision trees present in the random forest
max_depth = [10, 15, 21, 22, 23]

# min_samples_lead parameter defines the number of samples from each subset that is neededto create a leaf node
min_samples_leaf = [2, 4, 6]

# min_samples_split parameter defines the number of samples from each subset needed to create a new condition node
min_samples_split = [2, 4, 6]

# class_weight parameter defines the weights of every possible output value
# model will pay more attention to classes with higher weights
# class weights can be defined manually, the same way as was the case with a single decision tree

# 'balanced': При использовании этой опции веса классов автоматически устанавливаются
# обратно пропорционально частотам классов во входных данных. Это означает, что
# классы с меньшим количеством примеров будут иметь больший вес, чтобы их вклад
# в обучение модели был пропорционально больше.

# The value 'balanced_subsample' creates weights for each decision tree separately based on
# the subsample of the dataset which is used for its training. Этот вариант работает
# аналогично balanced, но веса классов устанавливаются обратно пропорционально их
# частотам в каждой подвыборке (bootstrap sample) при обучении каждого дерева в ансамбле.
# В отличие от balanced, где веса классов устанавливаются один раз для всего набора
# обучающих данных, balanced_subsample пересчитывает веса для каждого отдельного дерева
class_weight = ['balanced', 'balanced_subsample']

params = {'n_estimators': n_estimators,
          'max_depth': max_depth,
          'min_samples_leaf': min_samples_leaf,
          'min_samples_split': min_samples_split,
          'class_weight': class_weight}

optimizer = GridSearchCV(RandomForestClassifier(), params, cv=5)
optimizer.fit(X_train, y_train)
predicted = optimizer.predict(X_test)
accuracy = evaluate_accuracy(list(y_test), predicted)
print(f'\nClassification accuracy after grid search: {accuracy}\n')

params = optimizer.best_params_
print(f'The best results were obtained using the following set of hyperparameters: \n{params}\n\n')

###################################################################################################################

# conduct a grid search for the winequality_white dataset with different combinations of hyperparameters
# find the best parameters for classification
# compare the achieved results with the results obtained before optimization

###################################################################################################################
