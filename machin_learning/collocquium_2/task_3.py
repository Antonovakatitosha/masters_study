import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

length = 500
clusters = 3

points = np.zeros([length, 2])
for i in range(length):
    points[i] = [random.random(), random.random()]

plt.figure()
plt.scatter(points[:, 0], points[:, 1])
plt.show()

KM = KMeans(clusters)
KM.fit(points)
predicted = KM.predict(points)
# print(predicted)

data = pd.DataFrame(columns=['x', 'y', 'cluster'])
data['x'] = points[:, 0]
data['y'] = points[:, 1]
data['cluster'] = predicted
groups = data.groupby('cluster')

plt.figure()
plt.title('K-Means')
for name, group in groups:
    # print(name, '///', group)  # name - 0, 1, 2; group - table (x  y  cluster)
    plt.scatter(group.x, group.y)
plt.show()

