import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def knn_predict(train, labels, point, k):
    distance = []
    for i in range(len(train)):
        dist = euclidean_distance(point, train[i])
        distance.append((dist, labels[i]))
    distance.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distance[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]


# Test sample 1
###############
train = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
labels = ['A', 'A', 'A', 'B', 'B']
point = [4, 5]
k = 3

prediction = knn_predict(train, labels, point, k)
print('Class for the point {}, is: {}'.format(point, prediction))



#Test sample 2
##############
train = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
labels = ['A', 'A', 'A', 'B', 'B']
point = [5, 6]
k = 3

prediction = knn_predict(train, labels, point, k)
print('Class for the point {}, is: {}'.format(point, prediction))