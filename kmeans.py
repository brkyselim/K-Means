import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset(limit=None):
    #https://www.kaggle.com/luisfredgs/kmeans-tutorial
    df = pd.read_csv('dataset.csv')
    data = df.iloc[:, [3,4]].values 
    if limit is not None:
       data = data[:limit]
    scaler = StandardScaler() 
    data = scaler.fit_transform(data)
    return data


def show_dataset():
    dataset = load_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:, 0], dataset[:, 1])
    plt.title("Dataset")
    plt.show()

def euclidean_distance(a, b):
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist

def initial_centroids(dataset, k):
    # ilk işlemlerden birisi olarak rastgele merkezler seçiyoruz, bunlar üzerinden işlem yapıp yeni atamalar yapacağız.
    dataset = list(dataset)
    centroids = random.sample(dataset, k)
    return centroids

def min_distance(dataset, centroids):
    cluster = dict()
    k = len(centroids)
    for item in dataset:
        a = item
        flag = -1
        min_dist = float("inf") 
        for i in range(k):
            b = centroids[i]
            dist = euclidean_distance(a, b)
            if dist < min_dist:
                min_dist = dist
                flag = i
        if flag not in cluster.keys():
            cluster[flag] = []
        cluster[flag].append(item)
    return cluster


def reassign_centroids(cluster):
    centroids = []
    for key in cluster.keys():
        centroid = np.mean(cluster[key], axis=0)
        centroids.append(centroid)
    return centroids


def closeness(centroids, cluster):
    sum_dist = 0.0
    for key in cluster.keys():
        a = centroids[key]
        dist = 0.0
        for item in cluster[key]:
            b = item
            dist += euclidean_distance(a, b)
        sum_dist += dist
    return sum_dist

def show_cluster(centroids, cluster):
    cluster_color  = ['or', 'ob', 'og', 'ok', 'oy', 'ow']
    centroid_color = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

    for key in cluster.keys():
        plt.plot(centroids[key][0], centroids[key][1], centroid_color[key], marker='x')
        for item in cluster[key]:
            plt.plot(item[0], item[1], cluster_color[key])
    plt.title("K-Means")
    plt.show()

def k_means(k):
    dataset = load_dataset()
    centroids = initial_centroids(dataset, k)
    cluster = min_distance(dataset, centroids)
    current_dist = closeness(centroids, cluster)
    old_dist = 0

    while abs(current_dist - old_dist) >= 0.00001:
        centroids = reassign_centroids(cluster)
        cluster = min_distance(dataset, centroids)
        old_dist = current_dist
        current_dist = closeness(centroids, cluster)
    return centroids, cluster

show_dataset()
k = 5
centroids, cluster = k_means(k)
show_cluster(centroids, cluster)