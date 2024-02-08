import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

data = np.array([[0, 0, 0],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 0]]).astype(int)

x_train = data[:, :-1]
y_train = data[:, -1]
n_features = x_train.shape[1]

print(n_features)

model = Sequential()
model.add(Dense(1, input_dim=n_features, activation='linear'))
model.compile(loss='mse', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=1, epochs=10, verbose=2)

acc = history.history['accuracy']
loss = history.history['loss']

print(acc, loss)


def get_color(predicted):
    return 'b' if predicted > 0 else 'r'


area = 100
fig = plt.figure()
plt.title('XOR', fontsize=20)
ax = fig.add_subplot()

#  red – 0 and  blue – 1.
ax.scatter(0, 0, s=area, c=get_color(model.predict([[0, 0]])), label="Class 0")
ax.scatter(0, 1, s=area, c=get_color(model.predict([[0, 1]])), label="Class 1")
ax.scatter(1, 0, s=area, c=get_color(model.predict([[1, 0]])), label="Class 1")
ax.scatter(1, 1, s=area, c=get_color(model.predict([[1, 1]])), label="Class 0")
plt.grid()
plt.show()

fig = plt.figure()
plt.title('XOR', fontsize=20)
ax = fig.add_subplot()
ax.scatter(0, 0, s=area, c='b', label="Class 0")
ax.scatter(0, 1, s=area, c='r', label="Class 1")
ax.scatter(1, 0, s=area, c='r', label="Class 1")
ax.scatter(1, 1, s=area, c='b', label="Class 0")
plt.grid()
plt.show()







