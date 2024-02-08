import pandas as pd


# Definition of the starting dataset
data = pd.DataFrame(columns=['weather', 'temperature', 'playBasketball'])

weather = ['sunny', 'sunny', 'cloudy', 'rainy', 'rainy', 'rainy', 'cloudy', 'sunny', 'sunny', 'rainy', 'sunny',
           'cloudy', 'cloudy', 'rainy']
temperature = ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot',
               'mild']
playBasketball = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

data['weather'] = weather
data['playBasketball'] = playBasketball
data['temperature'] = temperature


def NaiveBayes(sample):

    # This part of the code will not work because there are combinations of weather and temperature values which do not exist in the dataset
    # the combination rainy|hot will produce a temporary set of length 0 which will in turn produce a division by 0 error

    # temp = data.loc[(data['weather'] == sample['weather']) &
    #               (data['temperature']==sample['temperature'])]
    # p_yx_true = len(temp.loc[temp['playBasketball'] == 'yes']) / len(temp)
    # p_yx_false = len(temp.loc[temp['playBasketball'] == 'no']) / len(temp)

    # In order to avoid this situation the smoothing alpha parameter is introduced
    # alpha can be any number (usually a low value number)

    alpha = 1

    # It is also required to define the dim variable which represents the dimensionality of
    # the training dataset, not including the output columns. (the number dim is equal to
    # the number of columns of the dataset which are used as inputs for classification)
    dim = len(test.columns)  # dim = 2

    temp = data.loc[(data['weather'] == sample['weather']) &
                    (data['temperature'] == sample['temperature'])]
    p_yx_true = (alpha + len(temp.loc[temp['playBasketball'] == 'yes'])) / (dim * alpha + len(temp))
    p_yx_false = (alpha + len(temp.loc[temp['playBasketball'] == 'no'])) / (dim * alpha + len(temp))

    # The final decision is predicated upon if p(yes|x) is greater than p(no|x)
    return 'yes' if p_yx_true > p_yx_false else 'no'


# Definition of data samples for algorithm testing
test = pd.DataFrame(columns=data.columns[:-1])
test['weather'] = ['sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'cloudy', 'sunny', 'rainy', 'cloudy']
test['temperature'] = ['hot', 'hot', 'hot', 'mild', 'mild', 'mild', 'cool', 'cool', 'cool']

# Naive Bayes is used for each sample from the test set and the results are printed in the console
for i in range(len(test)):
    print('For input ' + test['weather'][i] + ', ' + test['temperature'][
        i] + ' Naive Bayes returns the prediction: ' + str(NaiveBayes(test.iloc[i])))
