from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

data = np.array([
    [10, 6, 'positive'],
    [20, 5, 'positive'],
    [5, 4, 'negative'],
    [2, 5, 'negative'],
    [2, 4, 'negative'],
    [3, 6, 'positive'],
    [10, 7, 'positive'],
    [15, 8, 'positive'],
    [5, 9, 'positive'],
])

X = data[:, :-1].astype(int)
y = LabelEncoder().fit_transform(data[:, -1])
n_features = X.shape[1]

print(X, y)


def create_model():
    model = Sequential()

    model.add(Dense(6, input_dim=n_features, activation='relu'))
    # model.add(Dense(1, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


model = create_model()
history = model.fit(X, y, batch_size=2, epochs=50)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.title('model_1')
plt.show()


def create_model_2():
    model = Sequential()

    model.add(Dense(5, input_dim=n_features, activation='relu'))
    model.add(Dense(4, input_dim=n_features, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


model = create_model_2()
model.fit(X, y, batch_size=2, epochs=50)
print(model.predict([[2, 3]]))




