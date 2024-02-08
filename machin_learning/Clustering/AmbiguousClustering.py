import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

length = 600

# we define points in space in the form of x concentric circles
# we observe the bahaviour of different algorithms when clustering is ambiguous
points = np.zeros([length, 2])
for i in range(length):
    r = i % 2 + 1
    angle = random.randint(0, 360)
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    points[i] = [x, y]

plt.figure()
plt.scatter(points[:, 0], points[:, 1])
plt.show()

KM = KMeans(2)
KM.fit(points)
predicted = KM.predict(points)

data = pd.DataFrame(columns=['x', 'y', 'cluster'])
data['x'] = points[:, 1]
data['y'] = points[:, 0]
data['cluster'] = predicted

plt.figure()
plt.title('K-Means clustering')
groups = data.groupby('cluster')
for name, group in groups:
    plt.scatter(group.x, group.y)
plt.show()

Birch = Birch()
Birch.fit(points)
predicted = Birch.predict(points)

data = pd.DataFrame(columns=['x', 'y', 'cluster'])
data['x'] = points[:, 1]
data['y'] = points[:, 0]
data['cluster'] = predicted

plt.figure()
plt.title('Birch clustering')
groups = data.groupby('cluster')
for name, group in groups:
    plt.scatter(group.x, group.y)
plt.show()

DB = DBSCAN()
DB.fit(points)
predicted = DB.fit_predict(points)

data = pd.DataFrame(columns=['x', 'y', 'cluster'])
data['x'] = points[:, 1]
data['y'] = points[:, 0]
data['cluster'] = predicted

plt.figure()
plt.title('DBSCAN clustering')
groups = data.groupby('cluster')
for name, group in groups:
    plt.scatter(group.x, group.y)
plt.show()