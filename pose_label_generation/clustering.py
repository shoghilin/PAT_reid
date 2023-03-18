import json, os, re
import os.path as osp
import argparse, itertools
import torch, random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from feature_extract import get_dataloader, feature_extraction

# Reproducibility
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def add_none_result(dataset, pose_labels, mode='new_label'):
    '''
    This function is about adding the sample with no pose result to the dataset.
    * mode : Different way of adding none pose. 
        ignore : ignore the pose with no pose detected.
        cam_labels : add the pose label with the camera label.
        new_label : add an additional pose label.
    '''
    if mode=='ignore': return pose_labels

    root=f"./pose_result/{dataset}_jsons"
    none_id = max(list(pose_labels.values()))

    # read pose estimation result from json    
    none_pose = {}
    for i in os.listdir(root):
        with open(osp.join(root, i), 'r') as f:
            info = json.load(f)
            people = info['people']

        if len(people) == 0:
            try:
                if mode == 'cam_labels':
                    none_pose[i.split("_k")[0]] = int(re.search('_c(\d)', i).group(1)) + none_id
                elif mode == 'new_label':
                    none_pose[i.split("_k")[0]] = 1 + none_id
            except:
                print(i)

    before = len(pose_labels.keys())
    pose_labels.update(none_pose)
    print(f"None label added!\nBefor : {before}, After : {len(pose_labels.keys())}")

    return pose_labels


def get_pose_label_from_each_cam_image(X, n_clusters):
    feature_dict = {}
    for i in range(len(fnames)):
        f = fnames[i]
        cam = re.search("_c(\d*)", f).group(1)
        feature_dict.setdefault(f"cam_{cam}", [[], []])
        feature_dict[f"cam_{cam}"][0].append(f)
        feature_dict[f"cam_{cam}"][1].append(X[i, :].reshape(1, -1))

    # Clustering for each camera
    c_keys = sorted(list(feature_dict.keys()))
    print(c_keys)
    pose_labels = {}
    for i, c_key in enumerate(c_keys):
        fname, features = feature_dict[c_key]
        X_cam = np.concatenate(features, axis=0)
        
        clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X_cam)

        labels = (clustering.labels_  + i * n_clusters).tolist()
        pose_labels.update(dict(zip(fname, labels)))
    return pose_labels
    

def get_pose_label(X, n_clusters):    
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return dict(zip(fnames, clustering.labels_.tolist()))    


def set_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["market", "duke", "market_test", "duke_test", "lab_combine", "msmt17"], default="market")
    parser.add_argument("--n_cluster", choices=[4, 8], type=int, default=8)
    parser.add_argument("--mode", choices=['each_cam', 'overall'], default='each_cam')
    parser.add_argument("--none_mode", choices=['ignore', 'cam_labels', 'new_label'], default='new_label')
    parser.add_argument("--save_json", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = set_config()
    filename = f"pose_labels_{args.dataset}.json"

    # load data and extract features
    dataloader = get_dataloader(name=args.dataset)
    fmaps, fnames = feature_extraction(dataloader=dataloader)


    if args.mode == 'each_cam':
        pose_labels = get_pose_label_from_each_cam_image(X=fmaps, n_clusters=args.n_cluster)
    elif args.mode == 'overall':
        pose_labels = get_pose_label(X=fmaps, n_clusters=args.n_cluster)


    pose_labels = add_none_result(args.dataset, pose_labels, mode=args.none_mode)

    if args.save_json:
        save_dir = f"pose_labels/{args.mode}-{args.none_mode}-{args.n_cluster}"
        os.makedirs(save_dir, exist_ok=True)
        with open(osp.join(save_dir, f"pose_labels_{args.dataset}.json"), "w") as f:
            json.dump(pose_labels, f, indent = 4)
        print(f"pose_labels_{args.dataset}.json -- Done")

