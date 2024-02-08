from numpy import loadtxt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
print(dataset)

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9]
df = pd.DataFrame(dataset, columns=column_names)

# pairplot
# sns.pairplot(data=df, hue=9)
# plt.savefig('pairplot')
# plt.show()

X = dataset[:, :-1].astype('float')
X = StandardScaler().fit_transform(X)
y = dataset[:, -1].astype('int')
n_features = X.shape[1]
n_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, stratify=y)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='winter', edgecolor='k', label='Train')
plt.title('Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='autumn', edgecolor='k', label='Test')
plt.title('Test Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

model = Sequential()
model.add(Dense(15, input_dim=n_features, activation='tanh'))
model.add(Dense(10, input_dim=n_features, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'],  optimizer='adam')
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
TP = matrix[0, 0]
TN = matrix[1, 1]
FP = matrix[0, 1]
FN = matrix[1, 0]
print(f"TP = {TP}\n"
      f"TN = {TN}\n"
      f"FP = {FP}\n"
      f"FN = {FN}\n")

precision = TP / (TP + FP)
sensitivity = TP / (TP + FN)
f1 = 2 * precision * sensitivity / (precision + sensitivity)
accuracy = np.trace(matrix)

print(f'precision = {precision}, sensitivity = {sensitivity}, f1 = {f1}, accuracy = {accuracy}')




