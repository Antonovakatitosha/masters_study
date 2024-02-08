import pandas as pd

weather = ['sunny', 'sunny', 'cloudy', 'rainy', 'rainy', 'rainy', 'cloudy', 'sunny', 'sunny', 'rainy', 'sunny',
           'cloudy', 'cloudy', 'rainy']
playBasketball = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

# Definition of the starting dataset
data = pd.DataFrame(columns=['weather', 'playBasketball'])
data['weather'] = weather
data['playBasketball'] = playBasketball


def NaiveBayes(sample):
    # Naive Bayes requires the values of p(y), p(x) and p(x|y) for y='yes' and y='no'
    # each of these values will be calculated and inserted into the equation
    # p(y|x) = p(x|y) * p(y) / p(x)

    # p(x|y) for y = yes   Когда погода солнечная, при условии, что игра возможна
    temp = data.loc[data['playBasketball'] == 'yes']
    p_xy_true = len(temp.loc[temp['weather'] == sample['weather']]) / len(temp)

    # p(x|y) for y = no   Когда погода солнечная, при условии, что игра не возможна
    temp = data.loc[data['playBasketball'] == 'no']
    p_xy_false = len(temp.loc[temp['weather'] == sample['weather']]) / len(temp)

    # p(y) for y = yes   Какая вероятность того, что играть можно
    temp = data.loc[data['playBasketball'] == 'yes']
    p_y_true = len(temp) / len(data)

    # p(y) for y = no   Какая вероятность того, что играть нельзя
    temp = data.loc[data['playBasketball'] == 'no']
    p_y_false = len(temp) / len(data)

    # p(x)  Какая вероятность того, что погода будет солнечная
    temp = data.loc[data['weather'] == sample['weather']]
    p_x = len(temp) / len(data)

    # The acquired values are used for calculating p(y|x)
    # Какая вероятность того, можно играть, когда солнечно
    p_yx_true = p_xy_true * p_y_true / p_x
    # Какая вероятность того, нельзя играть, когда солнечно
    p_yx_false = p_xy_false * p_y_false / p_x

    # The final decision is predicated upon if p(yes|x) is greater than p(no|x)
    return 'yes' if p_yx_true > p_yx_false else 'no'


# Definition of data samples for algorithm testing
test = pd.DataFrame(columns=data.columns[:-1])
test['weather'] = ['sunny']
# test['weather'] = ['sunny', 'rainy', 'cloudy']

# Naive Bayes is used for each sample from the test set and the results are printed in the console
for i in range(len(test)):
    print('For input ' + test['weather'][i] + ' Naive Bayes returns the prediction: ' + str(NaiveBayes(test.iloc[i])))

# ❗The final probabilities can be calculated without using the Bayesian theorem.
# Namely, the probability of each outcome can be calculated by counting the number of
# samples for which the condition x applies and then counting the number of
# instances in that subset for which each of the possible y values happened.
# The probability is achieved by dividing the probability of each outcome by
# the probability of the condition being applied.
temp = data.loc[data['weather'] == test['weather'][0]]
example_true = len(temp.loc[temp['playBasketball'] == 'yes']) / len(temp)
example_false = len(temp.loc[temp['playBasketball'] == 'no']) / len(temp)
print(f'example_true = {example_true}')
print(f'example_false = {example_false}')
