import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Loading the Iris dataset
iris = pd.read_csv('bezdekIris.csv')

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
