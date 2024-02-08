import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from sklearn.preprocessing import StandardScaler

energy_data = pd.read_csv('ENB2012_data.csv')
print(len(energy_data))

X = energy_data[energy_data.columns[:-2]]
Y1 = energy_data[energy_data.columns[-2]]
Y2 = energy_data[energy_data.columns[-1]]
x_train, x_test, y_train_1, y_test_1, y_train_2, y_test_2, = train_test_split(X, Y1, Y2, test_size=0.2)


def evaluate(prediction, target, name):
    squared_error = np.sum(np.power(target - prediction, 2))
    mean_squared_error = squared_error / len(target)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    print('\n' + name)
    print(f'mean squared error: {str(round(mean_squared_error, 2))}, '
          f'root mean squared error: {str(round(root_mean_squared_error, 2))}')


RL_1 = LinearRegression()
RL_1.fit(x_train, y_train_1)
prediction = RL_1.predict(x_test)
evaluate(prediction, y_test_1, 'Linear Regression for first parameter')
R2 = RL_1.score(X, Y1)
print(f'R2 for first parameter = {R2}')

RL_2 = LinearRegression()
RL_2.fit(x_train, y_train_2)
prediction = RL_2.predict(x_test)
evaluate(prediction, y_test_2, 'Linear Regression for second parameter')
R2 = RL_2.score(X, Y2)
print(f'R2 for second parameter = {R2}')

tree_1 = DecisionTreeRegressor()
tree_1.fit(x_train, y_train_1)
prediction = tree_1.predict(x_test)
evaluate(prediction, y_test_1, 'Decision Tree for first parameter')

tree_2 = DecisionTreeRegressor()
tree_2.fit(x_train, y_train_2)
prediction = tree_2.predict(x_test)
evaluate(prediction, y_test_2, 'Decision Tree for second parameter')

forest_1 = RandomForestRegressor()
forest_1.fit(x_train, y_train_1)
prediction = forest_1.predict(x_test)
evaluate(prediction, y_test_1, 'Random Forest for first parameter')

forest_2 = RandomForestRegressor()
forest_2.fit(x_train, y_train_2)
prediction = forest_2.predict(x_test)
evaluate(prediction, y_test_2, 'Random Forest for second parameter')

n_estimators = [5, 10, 12]
max_depth = [15, 17, 20, 22, 25]
min_samples_split = [2, 4, 6]
min_samples_leaf = [2, 4, 6]
params = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

forest_greed_1 = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5)
forest_greed_1.fit(x_train, y_train_1)
print(f'\nBest params for first forest is {forest_greed_1.best_params_}')
forest_greed_1 = forest_greed_1.best_estimator_
prediction = forest_greed_1.predict(x_test)
evaluate(prediction, y_test_1, 'Grid Search for first parameter')

forest_greed_2 = GridSearchCV(RandomForestRegressor(), param_grid=params, cv=5)
forest_greed_2.fit(x_train, y_train_2)
print(f'\nBest params for second forest is {forest_greed_2.best_params_}')
forest_greed_1 = forest_greed_2.best_estimator_
prediction = forest_greed_2.predict(x_test)
evaluate(prediction, y_test_2, 'Grid Search for second parameter')

############# neural network #####################

n_features = X.shape[1]
x_train = x_train.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
x_test = x_test.apply(lambda x: (x - x.min()) / (x.max() - x.min()))


def create_model(optimizer='adam', activation='relu', kernel_initializer='he_normal'):
    visible = Input(shape=(n_features, ))
    hidden1 = Dense(8, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=l1(0.02))(visible)
    hidden2 = Dense(10, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=l1(0.02))(hidden1)
    hidden3 = Dense(10, activation=activation, kernel_initializer=kernel_initializer)(hidden2)

    output1 = Dense(1, activation='linear')(hidden3)
    output2 = Dense(1, activation='linear')(hidden3)

    model = Model(inputs=visible, outputs=[output1, output2])
    model.compile(loss=['mse'], optimizer=optimizer)
    return model


model = create_model()
plot_model(model, to_file='model.png', show_shapes=True)

early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
model.fit(x_train, [y_train_1, y_train_2], epochs=250, batch_size=5, verbose=0, callbacks=[early_stopping])

prediction1, prediction2 = model.predict(x_test)
evaluate(prediction1.flatten(), y_test_1, 'Neural network for first parameter')
evaluate(prediction1.flatten(), y_test_1, 'Neural network for second parameter')

