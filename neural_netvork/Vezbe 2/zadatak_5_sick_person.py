
import random
import matplotlib.pyplot as plt


trainingset = [[1, 2, 1], [-1, 2, 0], [0, -1, 0]]
eta = 1
maxiterations = 100

w1 = 1
w2 = -0.8
b = 0
error = random.uniform(-0.2, 0.2)
count = 0


while count < maxiterations and error != 0:
    error = 0
    for array in trainingset:
        target = array[2]
        output = 0
        summation = w1*array[0] + w2*array[1] + b
        if(summation > 0):
            output = 1
        else:
            output = 0
            
        if(output != target):
            error += 1
            
        w1 += eta*(target - output)*array[0]
        w2 += eta*(target - output)*array[1]
        b += eta*(target - output)
        
        print("output " + str(output) + " target " + str(target))
        print("ERROR " + str(error))
    count += 1

print("COUNT " + str(count))
print("ENDING ERROR" + str(error))
print("w1 " + str(w1) + " w2 " + str(w2) + " b " + str(b))

# visualisation
area = 200
fig = plt.figure(figsize=(6, 6))
plt.title('Plot', fontsize=20)
# color red: is class 0 and color blue is class 1.
plt.scatter(trainingset[0][0],trainingset[0][1], s=area, c='r', label="Class 0")
plt.scatter(trainingset[1][0],trainingset[1][1], s=area, c='b', label="Class 0")
plt.scatter(trainingset[2][0],trainingset[2][1], s=area, c='b', label="Class 0")
plt.grid()
plt.show()