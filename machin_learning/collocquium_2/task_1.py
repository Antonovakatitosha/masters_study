import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


dataset = pd.read_csv('SeoulBikeData.csv', usecols=lambda column: column != 'Date')
# print(dataset.head())

for i in ['Seasons', 'Holiday', 'Functioning Day']:
    codes, uniques = pd.factorize(dataset[i])
    dataset[i] = codes
print(dataset.head())

X = dataset[dataset.columns[1:]]
y = dataset[dataset.columns[1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

LR = LinearRegression()
LR.fit(X_train, y_train)
prediction = LR.predict(X_test)
# print(predicted, np.array(y_test))

DecisionTree = RandomForestRegressor()
DecisionTree.fit(X_train, y_train)
predicted = DecisionTree.predict(X_test)


def evaluate(prediction, target):
    square_error = np.sum(np.power(np.array(target - prediction), 2))

    mean_square_error = square_error / len(prediction)
    root_mean_square_error = np.sqrt(mean_square_error)

    print(f'mse: {str(round(mean_square_error, 2))}, root mse: {str(round(root_mean_square_error, 2))}')


print('Linear Regression')
evaluate(prediction, y_test)
print('\nRandom Forest')
evaluate(predicted, y_test)

n_estimators = [10, 25, 50]
max_depth = [4, 6, 8, 10]
min_samples_leaf = [2, 4, 6]
min_samples_split = [2, 4, 6]
params = {'n_estimators': n_estimators,
          'max_depth': max_depth,
          'min_samples_leaf': min_samples_leaf,
          'min_samples_split': min_samples_split
          }

random_forest_grid = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=7)
random_forest_grid.fit(X_train, y_train)
model = random_forest_grid.best_estimator_

print('\nRandom Forest Grid Search')
predicted = model.predict(X_test)

evaluate(predicted, y_test)
print(f'best params: {random_forest_grid.best_params_}')

