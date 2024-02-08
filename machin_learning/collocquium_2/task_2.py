import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

dataset = pd.read_csv('zoo.csv')

codes, uniques = pd.factorize(dataset['name'])
dataset['name'] = codes
# print(dataset.head())

X = dataset[dataset.columns[:-1]]
y = dataset[dataset.columns[-1]]
number_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = X_train.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
X_test = X_test.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# print(X_train)

# print(np.unique(y))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
number_classes = y_train.shape[1]
print(number_classes)


def create_model(number_of_neurons):
    model = Sequential()
    model.add(Dense(number_of_neurons, input_dim=number_features, activation='relu'))
    model.add(Dense(number_classes, input_dim=number_features, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# model = create_model(20)
# model.fit(X_train, y_train, batch_size=10, epochs=30, validation_data=(X_test, y_test))
# print(model.evaluate(X_test, y_test))


def create_multilayer_model(number_of_hidden_layer, neurons_number=[20, 15, 10]):
    model = Sequential()
    if number_of_hidden_layer != len(neurons_number): print('Error in number_of_hidden_layer')

    for layer in range(number_of_hidden_layer):
        if layer == 0:
            model.add(Dense(neurons_number[layer], input_dim=number_features, activation='relu'))
        else: model.add(Dense(neurons_number[layer], activation='relu'))

    model.add(Dense(number_classes, input_dim=number_features, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_multilayer_model(3, [20, 10, 10])
history = model.fit(X_train, y_train, batch_size=10, epochs=20, validation_data=(X_test, y_test))
print(model.evaluate(X_test, y_test))


def build_confusion_matrix(predicted, target):
    matrix = np.zeros((number_classes, number_classes)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[i]][target[i]] += 1
    print(matrix)
    return matrix


prediction = np.argmax(model.predict(X_test), axis=1)
target = np.argmax(y_test, axis=1)
confusion_matrix = build_confusion_matrix(prediction, target)

# Calculations for the first class
TP = confusion_matrix[0, 0]
TN = confusion_matrix[1:, 1:].sum()
FP = confusion_matrix[0].sum() - TP
FN = confusion_matrix[:, 0].sum() - TP
print(TP, TN, FP, FN)

accuracy = np.trace(confusion_matrix)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(accuracy, precision, recall, f1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(['Train', 'Vadidation'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Vadidation'])
plt.show()


