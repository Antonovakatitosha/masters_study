# compare statistical imputation strategies for the horse colic dataset
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# print(dataframe.head())


data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# evaluate each strategy on the dataset
results = list()
strategies = ['mean', 'median', 'most_frequent', 'constant']

for s in strategies:
	# create the modeling pipeline
	pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
	# evaluate the model
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	# store results
	results.append(scores)
	print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
# plot model performance for comparison
print(results)
pyplot.boxplot(results, labels=strategies, showmeans=True)
pyplot.show()
# Ящик (Box): Представляет интерквартильный размах (IQR), который охватывает 50% данных, лежащих между 25-м (нижний квартиль) и 75-м (верхний квартиль) процентилями.
# Серединная линия (Median Line): Линия внутри ящика, обозначающая медиану набора данных.
# Усы (Whiskers): Линии, выходящие из ящика к самым нижним и самым верхним значениям в наборе данных, за исключением выбросов. Обычно они представляют значения, находящиеся в пределах 1.5 * IQR от нижнего и верхнего квартиля.
# Точки (Outliers): Отдельные точки за пределами усов, которые обозначают выбросы в данных.
# Зеленые треугольники (Mean): Если showmeans=True, то на графике будут отображаться зеленые треугольники, показывающие среднее значение данных в каждой группе.
