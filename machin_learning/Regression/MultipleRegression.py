import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('Real estate.csv')

X_train, X_test, y_train, y_test = train_test_split(data[data.columns[1:-1]], data[data.columns[-1]], test_size=0.2)

# multiple regression, that is, regression with multiple input columns
# is conducted in the same way as simple (1-dimensional regression)
LR = LinearRegression()
LR.fit(X_train, y_train)
predicted = LR.predict(X_test)


# metrics used for model assessment do are not tied to the dimensionality of the input matrix
def evaluate(y_pred, y_true, name):
    square_error = np.array(y_test - y_pred)
    square_error = np.power(square_error, 2)
    square_error = np.sum(square_error)

    mean_square_error = square_error / len(y_pred)

    root_mean_square_error = np.sqrt(mean_square_error)

    print(name)
    print('mean square error: ' + str(round(mean_square_error, 2)))
    print('root mean square error: ' + str(round(root_mean_square_error, 2)))
    print()

    return mean_square_error, root_mean_square_error


mse, rmse = evaluate(predicted, list(y_test), 'linear regression')

# decision tree algorithm can be used for regression in the same way as classification
# regression using the decision tree algorithm is conducted in the same way as classification with an unknown number of classes
# the final deicision, in contrast to linear and polinomial regression is discrete and cannot produce values that are not a part of the training set
# the use of the deicsion tree for regression purposes is limited through conditions imposed by the dataset
# decision tree is not good in situation in which the scope of the possible output values is broad and the data is sparse
# decision tree is not good in situations in which the scope of outputs is not descriptive enough (it will produce problems with data extrapolation)
DecisionTree = DecisionTreeRegressor()
DecisionTree.fit(X_train, y_train)
predicted = DecisionTree.predict(X_test)
evaluate(predicted, y_test, 'decision tree regressor')

# the random forest algorithm solves problems with the decision tree, but increases the complexity of the model
# random forest can predict values that are not present in the starting dataset, however the outputs are still discrete
RandomForest = RandomForestRegressor()
RandomForest.fit(X_train, y_train)
predicted = RandomForest.predict(X_test)
evaluate(predicted, y_test, 'random forest regressor')

n_estimators = [25, 50, 75]
max_depth = [3, 5, 7, 9]
min_samples_leaf = [2, 4, 6]
min_samples_split = [2, 4, 6]
params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
          'min_samples_split': min_samples_split}

# grid search is used to try and improve the model in the same way that it is used for classification
RandomForestGrid = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5)
RandomForestGrid.fit(X_train, y_train)
model = RandomForestGrid.best_estimator_
predicted = model.predict(X_test)
evaluate(predicted, y_test, 'random forest regressor after grid search')
print('best params after grid search: ')
print(RandomForestGrid.best_params_)

# regression visualization of multidimensional data (where columns do not have the same root value) is difficult and complex
# in order to visualize the regression it is required that one or two input values are isolated and that inputs X are sorted in ascending order
# in order for the final function to be a sirection
