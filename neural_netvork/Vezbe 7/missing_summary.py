# summarize the horse colic dataset
from pandas import read_csv


url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')

# summarize the first few rows
print(dataframe.head())

# summarize the number of rows with missing values for each column
for i in range(dataframe.shape[1]):
    n_miss = dataframe[[i]].isnull().sum()
    perc = n_miss / dataframe.shape[0] * 100
    print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
