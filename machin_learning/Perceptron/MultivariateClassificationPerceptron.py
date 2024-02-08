import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('bezdekIris.csv')

data['class'] = data['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

X_train, X_test, y_train, y_test = train_test_split(data[data.columns[:-1]], data[data.columns[-1]], test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)


def createModel(neurons):
    model = Sequential()

    model.add(Dense(X_train.shape[1], input_shape=(X_train.shape[1],), activation='relu'))
    for i in neurons:
        model.add(Dense(i, activation='relu'))

    # because there are multiple output classes, we use the softmax activation function and the output layer has a number of neurons equal to the number of different classes
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    return model

# durign the creation of the model it is possible to dinamically assign parameter values which will change the model creation procedure
# in this case we pass the number of hidden layer and the number of neurons in those layers
# this approach is usefull because we do not have to change the entire model creation function, only the parameters which are passed

neurons = [10, 20, 30, 20, 10]
model = createModel(neurons)

# the softmax activation function of the output layer corrseponds to the categorical_crossentropy and sparse_categorical_crossentropy loss functions
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=300, batch_size=4)
prediction = model.predict(X_test)


# because of the categorical nature of our prediction, the output data will not have an appropriate shape for comparison with expected values
# cateogircal outputs will be of the shape (number of samples)x(number of output classes)
# in order to turn the output data into a 1-dimensional array we use the np.argmax command
# this command will return the index of the max value inside of an array which in this case represents the class with the highest degree of certainty
def categoricalPrediction(prediction):
    categorical = []
    for i in prediction:
        categorical.append(np.argmax(i))
    return categorical


prediction = categoricalPrediction(prediction)


# the multivariate classification evaluation procedure does not differ from the binary classification evaluation procedure
def confusionMatrix(y_true, y_pred):
    cm = confusion_matrix(y_pred, y_true)
    plt.figure()
    sb.heatmap(cm, annot=True, fmt='0')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')

    report = classification_report(y_true, y_pred)
    print(report)

    return cm


cm = confusionMatrix(list(y_test), prediction)


def modelHistory(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


modelHistory(history)
