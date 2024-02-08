import random

train_samples = [[0, 1, 1, 1],
                 [1, 1, 0, 1],
                 [1, 0, 1, 1],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [1, 0, 0, 0]]

test_samples = [[1, 1, 1, 1],
                [0, 0, 0, 0]]

random.shuffle(train_samples)

eta = 0.3
w1 = random.uniform(-0.2, 0.2)
w2 = random.uniform(-0.2, 0.2)
w3 = random.uniform(-0.2, 0.2)
b = random.uniform(-0.2, 0.2)
error = 1

epoch = 0

while epoch < 100 and error:
    error = 0

    for sample in train_samples:

        predicted = sample[0] * w1 + sample[1] * w2 + sample[2] * w3 + b
        predicted = int(predicted > 0)
        target = sample[3]

        if predicted != target:
            error += 1

            w1 += eta * (target - predicted) * sample[0]
            w2 += eta * (target - predicted) * sample[1]
            w3 += eta * (target - predicted) * sample[2]
            b += eta * (target - predicted)

    print(f"errors {error}")
    epoch += 1

print(f"epoch {epoch}")

test_1 = test_samples[0][0] * w1 + test_samples[0][1] * w2 + test_samples[0][2] * w3 + b
test_2 = test_samples[1][0] * w1 + test_samples[1][1] * w2 + test_samples[1][2] * w3 + b

print(f"TEST_1 {int(test_1 > 0) == test_samples[0][3]}")
print(f"TEST_2 {int(test_2 > 0) == test_samples[1][3]}")

