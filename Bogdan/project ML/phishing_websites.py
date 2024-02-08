import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

data_file_path = 'Training Dataset.arff'

with open(data_file_path, 'r') as file:
    for line_number, line in enumerate(file):
        if line.strip().lower() == '@data':
            data_start_line = line_number + 1
            break

phishing_data = pd.read_csv(data_file_path, skiprows=data_start_line, header=None)
# print(phishing_data.head(5))

X = phishing_data[phishing_data.columns[:-1]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
y = phishing_data[phishing_data.columns[-1]].replace(-1, 0)

number_features = X.shape[1]
number_class = len(np.unique(phishing_data[phishing_data.columns[-1]]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)


################################## Gaussian ####################################
def evaluate(prediction, target):
    counter = 0
    for i in range(len(prediction)):
        if prediction[i] == target[i]: counter += 1

    return round(counter / len(target), 2)


GNB = GaussianNB()
GNB.fit(X_train, y_train)
prediction = GNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Gaussian Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))

MNB = MultinomialNB()
MNB.fit(X_train, y_train)
prediction = MNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Multinomial Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))

BNB = BernoulliNB()
BNB.fit(X_train, y_train)
prediction = BNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Bernoulli Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))

CNB = CategoricalNB()
CNB.fit(X_train, y_train)
prediction = CNB.predict(X_test)
accuracy = evaluate(prediction, list(y_test))
print('Categorical Naive Bayes achieved a prediction accuracy of: ' + str(accuracy))


############################## METRICS ###############################
def build_confusion_matrix(predicted, target):
    matrix = np.zeros((number_class, number_class)).astype('int64')
    for i in range(len(predicted)):
        matrix[predicted[i]][target[i]] += 1
    return matrix


def precision_recall_f1(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)
    return round(precision, 2), round(recall, 2), round(F1_score, 2)


def calculate_metrics(method, predicted, target):
    print(f'\n {method}')
    confusion_matrix = build_confusion_matrix(predicted, target)

    accuracy = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    print(f"accuracy = {round(accuracy, 2)}")

    TP_1 = confusion_matrix[0, 0]
    TN_1 = confusion_matrix[1, 1]
    FP_1 = confusion_matrix[0, 1]
    FN_1 = confusion_matrix[1, 0]

    precision, recall, F1_score = precision_recall_f1(TP_1, TN_1, FP_1, FN_1)
    print(f"For first class: precision = {precision}, recall = {recall}, F1_score = {F1_score}")

    TP_2 = confusion_matrix[1, 1]
    TN_2 = confusion_matrix[0, 0]
    FP_2 = confusion_matrix[1, 0]
    FN_2 = confusion_matrix[0, 1]

    precision, recall, F1_score = precision_recall_f1(TP_2, TN_2, FP_2, FN_2)
    print(f"For second class: precision = {precision}, recall = {recall}, F1_score = {F1_score}")
########################################################################


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
predicted = decision_tree.predict(X_test)
calculate_metrics('Decision Tree', predicted, np.array(y_test))

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
predicted = random_forest.predict(X_test)
calculate_metrics('Random Forest', predicted, np.array(y_test))

n_estimators = [32, 115, 128, 145]
max_depth = [10, 15, 21, 22, 23]
min_samples_leaf = [2, 4, 6]
min_samples_split = [2, 4, 6]
class_weight = ['balanced', 'balanced_subsample']
params = {'n_estimators': n_estimators,
          'max_depth': max_depth,
          'min_samples_leaf': min_samples_leaf,
          'min_samples_split': min_samples_split,
          'class_weight': class_weight}

# optimizer = GridSearchCV(RandomForestClassifier(), params, cv=5)
# optimizer.fit(X_train, y_train)
# predicted = optimizer.predict(X_test)
# calculate_metrics('Grid Search', predicted, np.array(y_test))
# print(f'The best results were obtained using the following set of hyperparameters: \n{optimizer.best_params_}')

############################## Neural network ###############################


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


batch_size = 50
epochs = 350

model = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Calculation of class weights
class_weight = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(class_weight))  # Converting weights to a dictionary
print(f'\nclass_weight =  {class_weight}')

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                    callbacks=[early_stopping], class_weight=class_weight, verbose=0)

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

prediction = model.predict(X_test)
calculate_metrics('Neural network', predicted, np.array(y_test))


