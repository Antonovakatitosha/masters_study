import random

training_data = [[-0.5, -0.5, 1],
                 [-0.5, 0.5, 1],
                 [0.3, -0.5, 0],
                 [-0.1, -1, 0]]

# random.shuffle(training_data)
test_date = [0.7, -1.2]

eta = 0.3
w1 = random.uniform(-0.2, 0.2)
w2 = random.uniform(-0.2, 0.2)
b = random.uniform(-0.2, 0.2)
error = True

while error:
    error = False
    for i in range(len(training_data)):
        sample = training_data[i]

        predicted = sample[0] * w1 + sample[1] * w2 + b
        predicted = int(predicted > 0)
        target = sample[2]

        if target != predicted:
            error = True
            print("Adjust model")

            w1 += eta * (target - predicted) * sample[0]
            w2 += eta * (target - predicted) * sample[1]
            b += eta * (target - predicted)

        print(f"sample {sample}, result {int(sample[0] * w1 + sample[1] * w2 + b > 0)}")
    print()

print(f"sample {test_date}, result {int(test_date[0] * w1 + test_date[1] * w2 + b > 0)}")


