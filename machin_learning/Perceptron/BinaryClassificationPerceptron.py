import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

data = pd.read_csv('tic-tac-toe.csv')

inputs = data.columns[:-1]
outputs = data.columns[-1]

for i in inputs:
    data[i] = data[i].map({'o': 0, 'x': 1, 'b': 2})

data[outputs] = data[outputs].map({'negative': 0, 'positive': 1})

X_train, X_test, y_train, y_test = train_test_split(data[inputs], data[outputs], test_size=0.2)
# the training set is further divided into a training and validation subset so that the model progress could be followed during the training process throigh epochs
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)


# function for crating sequential models
def createModel():
    # Sequential model is a feed forward neural network and it represents the basic implementation of a perceptron in the keras library
    model = Sequential()
    # a perceptron contains an arbitrary number of dense layers
    # the first layer must have a defined input_shape which corresponds to the number of input features
    # the last layer is mandatory and is used for the final decision during classification
    # the number of hidden layers between the input and the output layer is arbitrary and can improve or worsen the classification

    model.add(Dense(len(inputs), input_shape=(len(inputs),), activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))

    # the output layer uses a sigmoid activation function due to the binary nature of the problem
    model.add(Dense(1, activation='sigmoid'))

    return model


model = createModel()

# the model needs to be compiled before training
# during the compilation of the model it is mandatory to define the optimizer and the loss function as well as the observed metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# class weights can be defined during training and they will impact the models bias towards defined classes

class_weight = {0: 4, 1: 1}
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=200,
                    batch_size=16, class_weight=class_weight)
prediction = model.predict(X_test)


# sigmoid function maps values between 0 and 1 so it is necessary for them to be binarized
def binaryPrediction(prediction, threshold):
    for i in range(len(prediction)):
        if prediction[i] >= threshold:
            prediction[i] = 1
        else:
            prediction[i] = 0
    return prediction


prediction = binaryPrediction(prediction, 0.6)


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


# it is possible to extract the following acquired information from the history variable
# loss function values of the chosen loss function
# values of all metrics defined in the evaluation array
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

# neural networks can be saved using the .save command in a .keras or .h5 file format
# saved neural networks can be used later for predicting new data without the need for retraining

# model.save('tic_tac_toe_model.h5')
