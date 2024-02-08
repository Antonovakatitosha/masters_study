import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
print(tf.__version__)
from scikeras.wrappers import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# load dataset
dataframe = pd.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)


# print(dummy_y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(model=baseline_model, epochs=200, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True)

# Это только оценка модели, она учится несколько раз, но данные не сохраняются, это нужно, если нет
# нужного количества тестовых данных. ПОСЛЕ ОЦЕНКИ МОДЕЛЬ НУЖНО УЧИТЬ ОТДЕЛЬНО
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# Обучение estimator на всех данных
estimator.fit(X, dummy_y)

# Предсказание на тестовых данных
# .reshape(1, -1): Этот метод изменяет форму массива так, чтобы он стал двумерным.
# Параметр 1 указывает, что в новом массиве будет одна строка, а -1 говорит NumPy
# автоматически вычислить нужное количество столбцов, чтобы сохранить все элементы массива.
# В результате одномерный массив превращается в двумерный массив формы (1, n).
# Это необходимо, потому что метод predict ожидает на вход двумерный массив,
# где каждая строка — это отдельный образец для предсказания.
Y_pred = estimator.predict(X[0].reshape(1, -1))

# Y_pred - предсказание в формате one-hot encoding
# Преобразуем one-hot encoding обратно в числовые метки
# axis=1 указывает функции np.argmax работать по столбцам. Для каждой строки
# (каждого примера) в массиве Y_pred функция np.argmax найдет индекс столбца,
# в котором находится максимальное значение. В контексте предсказаний модели
# это означает определение класса с наибольшей предсказанной вероятностью для каждого примера
predicted_classes = np.argmax(Y_pred, axis=1)
# Используем inverse_transform для преобразования числовых меток обратно в текстовые
predicted_labels = encoder.inverse_transform(predicted_classes)

print(predicted_labels, Y[0])

# Вывод матрицы ошибок и других метрик
print(confusion_matrix(dummy_y.argmax(axis=1), dummy_y.argmax(axis=1)))


