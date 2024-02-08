import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataset = pd.read_csv(url, header=None, na_values='?')

for i in range(dataset.shape[1]):
    quantity_of_missing = dataset[i].isnull().sum()
    percenteges = 100 * quantity_of_missing / dataset.shape[0]
    print(f'column is {i}, percentages is {round(quantity_of_missing, 2)}%')


# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp.fit(dataset)
# X_transformed = imp.transform(dataset)
# print(X_transformed)

new_dataset = SimpleImputer(strategy='mean').fit_transform(dataset)
# print(new_dataset)


def create_network():
    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


strategies = ["mean", "median", "most_frequent", "constant"]

for strategy in strategies:
    new_dataset = SimpleImputer(strategy=strategy).fit_transform(dataset.values)

    columns = [i for i in range(new_dataset.shape[1]) if i != 23]
    x = new_dataset[:, columns]
    y = new_dataset[:, 23].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = create_network()
    model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=0)
    error = model.evaluate(x_test, y_test)[1]
    print(f"for strategy {strategy}, accuracy: {round(error * 100, 2)}%")


# pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
# # evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)