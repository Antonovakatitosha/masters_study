import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

data = pd.read_csv('pima-indians-diabetes.csv', header=None)

# pairplot
sns.pairplot(data=data, hue=5)
plt.savefig('pairplot')
plt.show()

# correlation matrix
plt.figure(figsize=(8, 5))
sns.heatmap(data.corr(), annot=True, cmap='cubehelix_r')
plt.savefig('correlations')
plt.show()

X = data.values[:, :-1].astype('float')
X = StandardScaler().fit_transform(X)
y = data.values[:, -1].astype('int')

n_features = X.shape[1]
n_classes = len(np.unique(y))

# print(n_features, n_classes)
# print(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

model = Sequential()
model.add(Dense(20, input_dim=n_features, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(15, input_dim=n_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'],  optimizer='adam')
print(model.summary())
history = model.fit(X, y, batch_size=20, epochs=150, validation_data=(X_test, y_test))


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'])
plt.title('loss')
plt.show()

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'test'])
plt.title('accuracy')
plt.show()


def build_confusion_matrix(predicted, target):
    matrix = np.zeros((2, 2)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[1]][target[1]] += 1
    return matrix


matrix = build_confusion_matrix((model.predict(X_test) > 0.7).astype('int').flatten(), y_test)
print(f"TP = {matrix[0, 0]}\n"
      f"TN = {matrix[1, 1]}\n"
      f"FP = {matrix[0, 1]}\n"
      f"FN = {matrix[1, 0]}\n")

precision = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
sensitivity = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
f1 = 2 * precision * sensitivity/ (precision + sensitivity)
accuracy = (matrix[0, 0] + matrix[1, 1]) / (matrix[0, 0] + matrix[1, 1] + matrix[0, 1] + matrix[1, 0])
