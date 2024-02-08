import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

trucks_data = np.array([[10, 6, 'Truck'],
                        [20, 5, 'Truck'],
                        [5, 4, 'Van'],
                        [2, 5, 'Van'],
                        [3, 6, 'Truck'],
                        [10, 7, 'Truck'],
                        [15, 8, 'Truck'],
                        [5, 9, 'Truck']])

X = trucks_data[:, :-1].astype(int)
y = LabelEncoder().fit_transform(trucks_data[:, -1])
n_features = X.shape[1]
n_classes = len(np.unique(y))
print(y)

model_1 = Sequential()
# model_1.add(Dense(4, input_dim=n_features, activation='relu'))
model_1.add(Dense(4, input_dim=n_features, activation='tanh'))
# model_1.add(Dense(4, input_dim=n_features, activation='linear'))
# model_1.add(Dense(4, input_dim=n_features, activation='sigmoid'))

# model_1.add(Dense(n_classes, activation='softmax'))
# model_1.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_1.add(Dense(1, activation='sigmoid'))
model_1.compile(loss='binary_crossentropy', metrics=['accuracy'],  optimizer='adam')

print(model_1.summary())
history = model_1.fit(X, y, batch_size=2, epochs=60)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.title('model_1')
plt.show()

#______________________________________
model_2 = Sequential()
model_2.add(Dense(4, input_dim=n_features, activation='relu'))
model_2.add(Dense(3, activation='tanh'))
model_2.add(Dense(1, activation='sigmoid'))
model_2.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

print(model_2.summary())
history = model_2.fit(X, y, batch_size=2, epochs=60)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['loss', 'accuracy'])
plt.title('model_2')
plt.show()


def predict_type(prediction):
    return 'Truck' if prediction <= 0.5 else 'Van'


print(predict_type(model_1.predict([[2, 4]])))
print(predict_type(model_2.predict([[2, 4]])))

