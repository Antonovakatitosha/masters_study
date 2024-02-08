import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from matplotlib import cm
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from keras import backend as K
from imblearn.over_sampling import SMOTE
import seaborn
from keras.regularizers import l1, l2
from keras.utils import plot_model


data_file_path = 'tae.dat'

with open(data_file_path, 'r') as file:
    for line_number, line in enumerate(file):
        if line.strip().lower() == '@data':
            data_start_line = line_number + 1
            break

course_data = pd.read_csv(data_file_path, skiprows=data_start_line, header=None)
print(course_data.head(5))
course_data.info()

# Obtaining the amount of data in each class
class_counts = course_data[course_data.columns[-1]].value_counts()
print(f'amount of data in each class {class_counts}')



#################################### PLOTS #####################################

# pairplot
# sns.pairplot(data=course_data, hue=5)
# plt.savefig('Cource data, pairplot')

# plot Native depending on Course
# fig = course_data[course_data[5] == 1].plot(kind='scatter', x=0, y=3, color='red',label=1)
# course_data[course_data[5] == 2].plot(kind='scatter', x=0, y=3, color='green', label=2, ax=fig)
# course_data[course_data[5] == 3].plot(kind='scatter', x=0, y=3, color='yellow', label=3, ax=fig)
# fig.set_xlabel("Native")
# fig.set_ylabel("Course")
# fig.set_title("Native depending on Course")
# fig = plt.gcf()
# fig.set_size_inches(8, 5)
# plt.savefig('Native depending on Course')

# Creating ящичковой диаграммы (boxplots)
# plt.figure(figsize=(10, 6))
# sns.boxplot(x=course_data[5], y=course_data[4], data=course_data)
# plt.title('Distribution classes size for the scores')
# plt.savefig('Distribution classes size for the scores')

# Andrews curve
# plt.subplots(figsize=(10, 8))
# cmap = cm.get_cmap('summer')
# plotting.andrews_curves(course_data, 5, colormap=cmap)
# plt.savefig('Andrews curve for scores criteria')

# correlation matrix
# plt.figure(figsize=(8, 5))
# sns.heatmap(course_data.corr(), annot=True, cmap='cubehelix_r')
# plt.savefig('Source data correlation')

#########################################################################


# X = course_data[course_data.columns[:-1]]
X = course_data[course_data.columns[:-1]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# X = StandardScaler().fit_transform(course_data[course_data.columns[:-1]])
Y = LabelBinarizer().fit_transform(course_data[course_data.columns[-1]])
number_features = X.shape[1]
number_class = len(np.unique(course_data[course_data.columns[-1]]))

# stratify=Y tells train_test_split save class proportions, based on Y labels
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=101,
                                                    stratify=course_data[course_data.columns[-1]])
x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=101,
                                                    stratify=y_test)


smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(x_train, y_train)
# Now X_resampled и y_resampled contains synthetically generated examples
# print(X_resampled, y_resampled)

# real data
original_data = np.array(x_train)
# Generating random noise
noise = np.random.normal(0, 0.11, original_data.shape)
# Add noise to data
augmented_x_train = original_data + noise

# Combining noisy data with the original data set
x_train_augmented = np.concatenate((x_train, augmented_x_train, X_resampled), axis=0)
# Label duplication for noisy data
y_train_augmented = np.concatenate((y_train, y_train, y_resampled), axis=0)
print(f"train: {len(x_train)}, augmented: {len(x_train_augmented)}, validation: {len(x_validation)}, test: {len(x_test)}")


# np.random.seed(7)

# Function to create model, required for KerasClassifier
def create_model(optimizer='Nadam', init='he_uniform', activation='relu', dropout=0.3,
                 loss='categorical_crossentropy', dense_number=2):
    model = Sequential()
    model.add(Dense(20, input_dim=number_features, kernel_initializer=init, activation=activation,
                    kernel_regularizer=l1(0.02)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    for _ in range(dense_number):
        model.add(Dense(15, kernel_initializer=init, activation=activation))

    model.add(Dense(10, kernel_initializer=init, activation=activation, kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout))
    model.add(Dense(10, kernel_initializer=init, activation=activation, kernel_regularizer=l2(0.01)))
    model.add(Dense(5, kernel_initializer=init, activation=activation))

    # model.add(Dense(number_class, kernel_initializer=init, activation='softmax'))
    model.add(Dense(number_class, kernel_initializer=init, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


############################### Grid Search ###############################

# model = KerasClassifier(model=create_model, verbose=1)
#
# # optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# # activations = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# # init = ['uniform', 'lecun_uniform', 'normal', 'orthogonal', 'zero', 'one', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# # epochs = [100, 350]
# # batches = [5, 10, 25, 40]
# # dropout = [0.2, 0.3, 0.4, 0.5]
# optimizers =['Nadam']
# activations = ['relu']
# init = ['he_uniform']
# epochs = [350]
# batches = [15]
# dropout = [0.2]
# dense_number = [0, 1, 2, 3]
#
# param_grid = dict(epochs=epochs,
#                   batch_size=batches,
#                   model__init=init,
#                   model__activation=activations,
#                   model__optimizer=optimizers,
#                   model__dropout=dropout,
#                   model__dense_number=dense_number)
# grid = GridSearchCV(cv=5, estimator=model, param_grid=param_grid, n_jobs=-1, verbose=3)
# grid_result = grid.fit(X, Y)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # using {'batch_size': 10, 'epochs': 350, 'model__activation': 'relu', 'model__dropout': 0.3, 'model__init': 'glorot_normal', 'model__optimizer': 'Nadam'}
#
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

###########################################################################

# Determination of the loss function taking into account class weights
# def weighted_categorical_crossentropy(weights):
#     weights = K.variable(weights)
#
#     def loss(y_true, y_pred):
#         y_true = K.cast(y_true, 'float32')  # Data type conversion
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         loss = y_true * K.log(y_pred) * weights
#         loss = -K.sum(loss, -1)
#         return loss
#
#     return loss


# # Calculation of class weights
class_weights = compute_class_weight('balanced', classes=np.unique(course_data[course_data.columns[-1]]),
                                     y=course_data[course_data.columns[-1]])
# Converting weights to a dictionary
class_weight_dict = dict(enumerate(class_weights))
class_weight_dict[0] = 0.7
class_weight_dict[1] = 1.4

print(f'class_weights: {class_weight_dict}')  # --> {0: 0.7, 1: 1.4, 2: 0.967948717948718}

# model = create_model(loss=weighted_categorical_crossentropy(class_weights))
model = create_model()
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True)

epochs = 350
batch_size = 20

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
history = model.fit(x_train_augmented, y_train_augmented, epochs=epochs, batch_size=batch_size,
                    validation_data=(x_validation, y_validation), callbacks=[early_stopping], class_weight=class_weight_dict)

model.save('my_model.h5')  # Saving model

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

X = np.concatenate((x_train_augmented, x_validation), axis=0)
Y = np.concatenate((y_train_augmented, y_validation), axis=0)
model.fit(X, Y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], class_weight=class_weight_dict)


def build_confusion_matrix(predicted, target):
    matrix = np.zeros((number_class, number_class)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[i]][target[i]] += 1
    print(matrix)
    return matrix


def draw_matrix(matrix):
    plt.figure()
    seaborn.heatmap(matrix, annot=True, fmt='d', cmap="Greens")
    plt.xlabel('real values')
    plt.ylabel('predicted')
    plt.title("Confusion matrix")
    plt.savefig("Confusion matrix")
    plt.show()


prediction = model.predict(x_test)

# argmax from NumPy returns indices of maximum values along the specified axis
confusion_matrix = build_confusion_matrix(np.argmax(prediction, axis=1), np.argmax(y_test, axis=1))
draw_matrix(confusion_matrix)
report = classification_report(np.argmax(prediction, axis=1), np.argmax(y_test, axis=1))
print(report)
# print(f'class_weights = {class_weight_dict}')
