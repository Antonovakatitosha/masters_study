import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from matplotlib import cm
import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1, l2
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight


data_file_path = 'adult.dat'

with open(data_file_path, 'r') as file:
    for line_number, line in enumerate(file):
        if line.strip().lower() == '@data':
            data_start_line = line_number + 1
            break

adult_data = pd.read_csv(data_file_path, skiprows=data_start_line, header=None)
print(adult_data.head(5))


# # pairplot
# sns.pairplot(data=adult_data, hue=14)
# plt.savefig('adult data, pairplot')
#
# # Creating ящичковой диаграммы (boxplots)
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=adult_data[14], y=adult_data[4], data=adult_data)
# plt.title('Distribution adult for the Education-num')
# plt.savefig('Distribution adult for the Education-num')


unique_values_workclass = adult_data[1].unique()
print(unique_values_workclass)

for i in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
    codes, uniques = pd.factorize(adult_data[i])
    adult_data[i] = codes

for i in [0, 2, 4, 10, 11, 12]:
    min = adult_data[i].min()
    max = adult_data[i].max()
    adult_data[i] = adult_data[i].apply(lambda x: (x - min) / (max - min))


print(adult_data.head(5))

# # Andrews curve
# plt.subplots(figsize=(10, 8))
# cmap = cm.get_cmap('summer')
# plotting.andrews_curves(adult_data, 14, colormap=cmap)
# plt.savefig('Andrews curve for adult_data')

X = adult_data[adult_data.columns[:-1]]
y = adult_data[adult_data.columns[-1]]

number_features = X.shape[1]
number_class = len(np.unique(y))
print(f"number_features = {number_features}, number_class = {number_class}")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101,
                                                    stratify=adult_data[adult_data.columns[-1]])
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=101,
                                                    stratify=y_test)

print(f"train: {len(x_train)}, validation: {len(x_validation)}, test: {len(x_test)}")


def create_model(optimizer='Adagrad', init='glorot_uniform', activation='relu', dropout=0.2):
    model = Sequential()
    model.add(Dense(20, input_dim=number_features, kernel_initializer=init, activation=activation,
                    kernel_regularizer=l1(0.02)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(15, kernel_initializer=init, activation=activation, kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout))
    model.add(Dense(10, kernel_initializer=init, activation=activation, kernel_regularizer=l2(0.01)))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


############################### Grid Search ###############################

# model = KerasClassifier(model=create_model, verbose=1)
#
# optimizers = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Nadam']
# activations = ['softmax',  'relu', 'tanh', 'sigmoid']
# init = ['uniform', 'normal', 'zero', 'one', 'glorot_uniform']
# dropout = [0.2, 0.4]
#
# epochs = [100, 250]
# batches = [25, 50, 70]
#
# param_grid = dict(epochs=epochs,
#                   model__init=init,
#                   batch_size=batches,
#                   model__activation=activations,
#                   model__optimizer=optimizers,
#                   model__dropout=dropout)
# grid = GridSearchCV(cv=5, estimator=model, param_grid=param_grid, n_jobs=-1, verbose=1)
# grid_result = grid.fit(X, y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Best: 0.833525 using {'batch_size': 50, 'epochs': 250, 'model__activation': 'relu', 'model__dropout': 0.2,
# 'model__init': 'glorot_uniform', 'model__optimizer': 'Adagrad'}
###########################################################################

batch_size = 50
epochs = 250

model = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Calculation of class weights
class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(class_weight))  # Converting weights to a dictionary
print(f'class_weight =  {class_weight}')

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_validation, y_validation),
                    callbacks=[early_stopping], class_weight=class_weight)
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

X = np.concatenate((x_train, x_validation), axis=0)
Y = np.concatenate((y_train, y_validation), axis=0)
model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], class_weight=class_weight)

prediction = model.predict(x_test)


def build_confusion_matrix(predicted, target):
    matrix = np.zeros((number_class, number_class)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[i]][target[i]] += 1
    print(matrix)
    return matrix


def draw_matrix(matrix):
    plt.figure()
    sns.heatmap(matrix, annot=True, fmt='d', cmap="Greens")
    plt.xlabel('real values')
    plt.ylabel('predicted')
    plt.title("Confusion matrix")
    plt.savefig("Confusion matrix")
    plt.show()


# argmax from NumPy returns indices of maximum values along the specified axis
prediction = (prediction >= 0.7).astype(int).flatten()
confusion_matrix = build_confusion_matrix(prediction, y_test.values)

draw_matrix(confusion_matrix)
report = classification_report(prediction, y_test)
print(report)


