import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# https://www.kaggle.com/code/nataliyazhovannik/iris-classification-with-deep-learning

# read data
iris = pd.read_csv("Iris.csv")
iris.head(5)
iris.info()

# plot
sns.pairplot(data=iris[iris.columns[1:6]], hue='Species')
# plt.show()

# more plotting
fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red',
                                               label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green',
                                             label='Versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='yellow',
                                            label='Virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal length depending on Width")
fig = plt.gcf()
fig.set_size_inches(8, 5)
# plt.show()

#  build a heatmap with input as the correlation matrix calculated by iris.corr()
# iris['Species'] = iris['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
# print(iris.corr())
# plt.figure(figsize=(8,5))
# sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r')
# plt.show()

# Создание ящичковой диаграммы
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=iris['Species'], y=iris['PetalLengthCm'], data=iris)
# plt.title('Распределение длины лепестков по видам ириса')
# plt.show()

# Andrews curve
plt.subplots(figsize = (10, 8))
from pandas import plotting
from matplotlib import cm

cmap = cm.get_cmap('summer')
plotting.andrews_curves(iris.drop("Id", axis=1), "Species", colormap=cmap)
plt.show()


# drop ID and perform normalisation
iris.drop('Id', axis=1, inplace=True)
df_norm = iris[iris.columns[0:4]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)

# code output
target = iris[['Species']].replace(iris['Species'].unique(), [0, 1, 2])
df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)

# preprocessing
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)

# train test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# fit model
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=120, validation_data=(x_test, y_test))

# plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(['Train', 'Vadidation'])
plt.show()

# plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['Train', 'Vadidation'])
plt.show()

# confusion matrix

# predict for test dataset
predict_train = model.predict(x_train)
predict_test = model.predict(x_test)

cm_train = confusion_matrix(y_train.argmax(axis=1), predict_train.argmax(axis=1))
print(cm_train)
print(classification_report(y_train.argmax(axis=1), predict_train.argmax(axis=1)))

cm_test = confusion_matrix(y_test.argmax(axis=1), predict_test.argmax(axis=1))
print(cm_test)
print(classification_report(y_test.argmax(axis=1), predict_test.argmax(axis=1)))

# plot confusion matrix
ax = sns.heatmap(cm_test, annot=True, cmap='Blues')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.set_xlim([-0.5, 3])
ax.set_ylim([-0.5, 3])

# Display the visualization of the Confusion Matrix.
plt.show()
