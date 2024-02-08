import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

# Разделение на тренировочные и тестовые данные
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_train, x_test = x_train / 255.0, x_test / 255.0

# Создаем маску для фильтрации
mask = (y_train == 5)

# Применяем маску к x_train и y_train
x_train_5 = x_train[mask]
y_train_5 = y_train[mask]
# Теперь x_train_5 содержит только изображения, соответствующие цифре 5

# plt.imshow(x_train_5[100], cmap='gray')  # cmap='gray' используется для вывода в градациях серого
# plt.show()

# Преобразование каждого изображения в x_train_5 в одномерный массив
x_train_5_flattened = np.array([i.flatten() for i in x_train_5])

num_features = x_train_5_flattened.shape[1]
num_neurons = 1

weights = np.zeros(num_features)
bias = 1
eta = 1


for train in x_train_5_flattened:
    delta_y = eta * np.sign(np.sum(weights*train) + bias)

    weights = weights + delta_y * train
    bias = bias + delta_y

# for train in x_train_5_flattened:
#     result = np.sign(np.sum(weights*train) + bias)
#     if result == 0: print(result)
correct_recognize_5 = 0
incorrect_recognize_5 = 0
correct_recognize_non_5 = 0
incorrect_recognize_non_5 = 0


for test, label in zip(x_test, y_test):
    test_flatten = test.flatten()
    result = np.sign(np.sum(weights * test_flatten) + bias)
    if result == 1 and label == 5:
        correct_recognize_5 += 1

    elif result == 1 and label != 5:
        incorrect_recognize_5 += 1
        # plt.imshow(test, cmap='gray')  # cmap='gray' используется для вывода в градациях серого
        # plt.show()
        # break

    elif result != 1 and label != 5:
        correct_recognize_non_5 += 1

    elif result != 1 and label == 5:
        incorrect_recognize_non_5 += 1

print(f'correct recognize 5 – {correct_recognize_5}\n'
      f'incorrect recognize 5 – {incorrect_recognize_5}\n'
      f'correct recognize non 5 – {correct_recognize_non_5}\n'
      f'incorrect recognize non 5 – {incorrect_recognize_non_5}')

