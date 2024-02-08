import random

training_set = [[0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1]]
eta = 0.3
max_iterations = 100

w1 = random.uniform(-0.2, 0.2)
w2 = random.uniform(-0.2, 0.2)
b = random.uniform(-0.2, 0.2)

error = 1
count = 0

while count < max_iterations and error:
    error = 0

    for array in training_set:
        target = array[2]
        summation = w1 * array[0] + w2 * array[1] + b
        output = int(summation > 0)

        if output != target:
            error += 1

        w1 += eta * (target - output) * array[0]
        w2 += eta * (target - output) * array[1]
        b += eta * (target - output)

        print(f"output {output}, target {target}")

    print(f"ERROR {error}")
    count += 1

print("COUNT " + str(count))
print("w1 " + str(w1) + " w2 " + str(w2) + " b " + str(b))
