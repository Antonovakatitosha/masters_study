import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# Loading the Iris dataset
# iris = pd.read_csv('bezdekIris.csv')
iris_data = load_iris()
# Создание DataFrame из данных
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
# Добавление столбца с метками классов
iris['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
print(iris)

# Splitting the data into training and test subsamples
X_train, X_test, y_train, y_test = train_test_split(iris[iris.columns[:-1]], iris[iris.columns[-1]], test_size=0.2)


# Accuracy calculation function. Accuracy is calculated as a ratio between the number of correct prediction and the number of total predictions.
def evaluate(prediction, target):
    counter = 0
    for i in range(len(prediction)):
        if prediction[i] == target[i]: counter += 1

    return counter / len(target)


# Gaussian Naive Bayes is a classifier which is most commonly used with continual data parameter values
# for which it can be speculated that they fall under the Gaussian distribution (normal distribution).
GNB = GaussianNB()
GNB.fit(X_train, y_train)
prediction = GNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Gaussian Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))

#  Multinomial Naive Bayes is a classifier most commonly used on discrete data problems
#  which contain a large number of possible parameter values.
MNB = MultinomialNB()
MNB.fit(X_train, y_train)
prediction = MNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Multinomial Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))

# Bernoulli Naive Bayes is a classifier which is used for data which fall under the
# Bernoulli distribution. This distribution is very rare in real world datasets.
BNB = BernoulliNB()
BNB.fit(X_train, y_train)
prediction = BNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Bernoulli Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))

#  Categorical Naive Bayes is a classifier used when working with predominantly discrete
#  data samples in cases in which the number of possible parameter values is not large enough
#  for multinomial Naive Bayes to be required. (example 1 and 2 are examples of categorical Naive Bayes)
CNB = CategoricalNB()
CNB.fit(X_train, y_train)
prediction = CNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Categorical Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))
