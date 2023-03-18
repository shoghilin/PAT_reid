import json
import numpy as np
import os, shutil
import os.path as osp

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def read_json(root, threshold=0.2):
    data_list = os.listdir(root)
    # # read pose estimation result from json
    # people = {}
    # for i in data_list:
    #     with open(osp.join(root, i), 'r') as f:
    #         info = json.load(f)
    #         people[i] = info['people']

    # # store pose for each samples
    # pose = {}
    # for k in people.keys():
    #     if len(people[k]) == 1:
    #         # pose[k] = np.array(people[k][0]['pose_keypoints_2d']).reshape(-1, 3)[:, :2]
            
    #         info = np.array(people[k][0]['pose_keypoints_2d']).reshape(-1, 3)
    #         info[info[:, 2] < threshold] = 0
    #         pose[k.split("_k")[0]] = info[:, :2]
    
    pose = {}
    for i in data_list:
        with open(osp.join(root, i), 'r') as f:
            info = json.load(f)
            people = info['people']
            
        if len(people) == 1:
            # pose[k] = np.array(people[k][0]['pose_keypoints_2d']).reshape(-1, 3)[:, :2]            
            info = np.array(people[0]['pose_keypoints_2d']).reshape(-1, 3)
            info[info[:, 2] < threshold] = 0
            pose[i.split("_k")[0]] = info[:, :2]

    print(f"Actual/Total : {len(pose)}/{len(data_list)}, loss number/ratio : {len(data_list) - len(pose)}/{(len(data_list) - len(pose)) / len(data_list)*100:.2f}%")
    return pose

def process(name, X, k=4, type='kmeans'):
    X_embedded = TSNE(n_components=2, init='random', random_state=0).fit_transform(X)
    
    if type == "kmeans":
        clustering = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = clustering.labels_
    elif type == "GMM":
        clustering = GaussianMixture(n_components=k, random_state=0).fit(X)
        labels = clustering.predict(X)
    elif type == "Hierarchical":
        clustering = AgglomerativeClustering(n_clusters=k).fit(X)
        labels = clustering.labels_

    plt.figure(figsize=(10,10))
    plt.title(name)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=60, c=labels[:])
    plt.savefig(name)
    plt.show()
