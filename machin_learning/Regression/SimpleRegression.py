import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('tvmarketing.csv')

X_train, X_test, y_train, y_test = train_test_split(data['TV'], data['Sales'], test_size=0.2)

# linear regression creates a linear model defined as y=k1x1+k2x+...+n where k1,k2... are function
# growth coefficients and n is the intercept in x=0
LR = LinearRegression()

# the model is trained by calling the fit function and passing the training inputs and outputs
LR.fit(np.array(X_train).reshape(-1, 1), y_train)

# values are predicted by calling the predict function and passing the test inputs
predicted = LR.predict(np.array(X_test).reshape(-1, 1))


# regression is most commonly evaluated using the mean_squared_error and root_mean_squared_error metrics
# mean squared error measures the mean squared distance between the expected outputs and the predicted outputs in the n-dimensional space
# this metric does not have a ceiling value and is interpreted on a problem by problem basis
# root mean squared error metric is used for easier interpretation
def evaluate(y_true, y_pred, name):
    squared_error = y_true - y_pred
    squared_error = np.power(squared_error, 2)
    squared_error = np.sum(squared_error)

    mean_squared_error = squared_error / len(y_true)

    root_mean_squared_error = np.sqrt(mean_squared_error)

    print('\n' + name)
    print('mean squared error: ' + str(round(mean_squared_error, 2)))
    print('root mean squared error: ' + str(round(root_mean_squared_error, 2)))
    return mean_squared_error, root_mean_squared_error


evaluate(list(y_test), predicted, 'linear_regression')

# R2 score (R^2 score) is used for assessing how well the input data describes the output data
# it can be used to make decision about which data columns will be used when regression is conducted
r2 = LR.score(data[data.columns[:-1]], data[data.columns[-1]])
print('r2 score of the linear model: ' + str(round(r2, 2)))

# results of one-dimensional regression can be visualized
# using the test inputs and outputs we draw a scatter plot which represents the data distribution
# using the test inputs and predicted values we draw a line plot which represents the prediction model in space
plt.figure()
plt.title('regression visualization')
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predicted)
plt.show()

# results can also be reconstructed by changing the values of coefficients and intercepts in the equation y=kx+n
plt.figure()
plt.title('linear model reconstruction')
plt.scatter(X_test, y_test, color='black')

coefficients = LR.coef_
intercept = LR.intercept_
k = coefficients
n = intercept

line_x = np.arange(min(X_train), 1.01 * max(X_train), (max(X_train) - min(X_train)) / 10)
line_y = k * line_x + n

plt.plot(line_x, line_y, color='red')
plt.show()

# polinomial regression requires the increase in the dimensionality of the input parameter matrix to the power of n
# the increse of the dimensionality to the third degree, of a one dimensional regression matrix will result in a 4 column matrix: x^0 x^1 x^2 and x^3
# if polinomial regression is to be used with a higher dimensionality matrix then combinations of parameters will also be created, for example x1x2 x1^2x2 x1x2^2
# column with the values of x^0 will not be used for regression analysis due to the fact that all input values will be equal to 1
poly = PolynomialFeatures(3)
polynomialFeatures = poly.fit_transform(np.array(data['TV']).reshape(-1, 1))
print(polynomialFeatures)

# column 0 is skipped due to the aforementioned reason of x0=1
newData = pd.DataFrame(data={'x1': polynomialFeatures[:, 1],
                             'x2': polynomialFeatures[:, 2],
                             # 'x3':polynomialFeatures[:,3],
                             'Sales': data['Sales']})
print(newData)

X_train, X_test, y_train, y_test = train_test_split(newData[newData.columns[:-1]], newData[newData.columns[-1]],
                                                    test_size=0.2)

LR = LinearRegression()
LR.fit(X_train, y_train)
predict = LR.predict(X_test)

evaluate(list(y_test), predicted, 'polynomial_regression')

r2 = LR.score(newData[newData.columns[:-1]], newData[newData.columns[-1]])
print('r2 score of the polinomial model: ' + str(round(r2, 2)))

plt.figure()
plt.title('polynomial regression visualization')
plt.scatter(X_test[X_test.columns[0]], y_test, color='black')

# even though the matrix in question is multidimensional, since all of the columns have common root value,
# visualization can be conducted through the use of the equation y=k1x1+k2x2+n
coefficients = LR.coef_
intercept = LR.intercept_
k1 = coefficients[0]
k2 = coefficients[1]
# k3=coefficients[2]
n = intercept

X = X_train[X_train.columns[0]]
line_x1 = np.arange(min(X), 1.01 * max(X), (max(X) - min(X)) / 10)
line_x2 = line_x1 ** 2
# line_x3=line_x1**3

line_y2 = k1 * line_x1 + k2 * line_x2 + n
# line_y3=k1*line_x1+k2*line_x2+k3*line_x3+n

plt.plot(line_x1, line_y2, color='red')
# plt.plot(line_x1,line_y3,color='green')
plt.show()
