import os, re, sys, json
sys.path.append('.')
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import manifold
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sskd import datasets
from sskd import models
from sskd.utils.data import transforms as T
from sskd.utils.data.preprocessor import Preprocessor
from sskd.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


data_dir = {
    "market1501":"G:/Github/Datasets/market1501/Market-1501-v15.09.15/bounding_box_test",
    "dukemtmc-reid":"G:/Github/Datasets/dukemtmc-reid/DukeMTMC-reID/bounding_box_test",
}

mode_types = {
    "PAT":'resnet50-pose-mmt/overall-new_label-8', 
    "PCAT":'resnet50-pose-mmt/each_cam-new_label-8',
    "CAT":'resnet50-cam-mmt',
    "MMT":'resnet50-mmt'
}
pose_dir = {
    "market1501":"./evaluate_visualize/pose_labels_market_test.json",
    "dukemtmc-reid":"./evaluate_visualize/pose_labels_duke_test.json",
}

def get_label(x):
    try:
        return int(re.search("(-?\d+)_c", x).group(1))
    except:
        print(x)

def get_cam(x):
    try:
        return int(re.search(r"-?\d+_c(\d+)", x).group(1))
    except:
        print(x)

def create_dataset(target, root):
    # fList = [i for i in os.listdir(root) if '.jpg' in i]
    with open(pose_dir[target], 'r') as f:
        poseid = json.load(f)
    fList = list(map(lambda x: x+'.jpg', poseid.keys()))
    df = pd.DataFrame({"fname":fList, "pid":list(map(get_label, fList)), "cid":list(map(get_cam, fList)), "pose":poseid.values()})
    print(df.head())
    df = df[df['pid']!=0]
    df = df[df['pid']!=-1]
    return df.to_numpy()

def setupDataLoader(dataset, root):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    transform = T.Compose([
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                normalizer
            ])
    test_loader = DataLoader(
            Preprocessor(dataset, root=root, transform=transform, base_pose=True),
            batch_size=200, num_workers=8,
            shuffle=False, pin_memory=True)
    return test_loader




def create_model_AND_extract_feature(checkpoint_path, test_loader):
    # build model
    model = models.create('resnet50', num_features=0, dropout=0, num_classes=1000).cuda()
    model = nn.DataParallel(model)
    initial_weights = load_checkpoint(checkpoint_path)
    copy_state_dict(initial_weights['state_dict'], model)
    print(f"The best mAP is {initial_weights['best_mAP']}")
    model.eval()    

    # feature extraction
    features, pid_list, cid_list, poseid_list = [], [], [], []
    with torch.no_grad():
        for imgs, _, pids, cids, poseid in test_loader:
            outputs = model(imgs.cuda())
            cid_list.append(cids)
            pid_list.append(pids)
            features.append(outputs)        
            poseid_list.append(poseid)      
    features = torch.concat(features, dim=0).cpu().numpy()
    pid_list = torch.concat(pid_list, dim=0).cpu().numpy()
    cid_list = torch.concat(cid_list, dim=0).cpu().numpy()
    poseid_list = torch.concat(poseid_list, dim=0).cpu().numpy()

    return features, pid_list, cid_list, poseid_list


def run(source, target, root, weight_root):
    print(f"Start process task {source}->{target}")
    # prepare data (create dataset & setup loader)
    dataset = create_dataset(target, root)
    test_loader = setupDataLoader(dataset, root)    

    for mode in ["MMT", "PAT", "CAT", "PCAT"]:
        print(f"Current mode is {mode}")
        checkpoint_path=f"{weight_root}/{source}TO{target}/{mode_types[mode]}/model_best.pth.tar"

        # create model, extract feature, and reduce dimention to 2d
        features, pid_list, cid_list, poseid_list = create_model_AND_extract_feature(checkpoint_path, test_loader)

        # compute intra-class variance
        intra_var = 0
        id_center = [] # for compute inter-class variance
        for id in np.sort(np.unique(pid_list)):
            id_center.append(features[pid_list==id, :].mean(axis=0))
            intra_var += features[pid_list==id, :].var(axis=0).sum()
        id_center = np.array(id_center)
        print("Intra-class variance :", intra_var/id_center.shape[0])

        # compute inter-class variance
        inter_var = id_center.var(axis=0).sum()
        print("Inter-class variance :", inter_var)
        
        # compute camera variance
        cam_var = 0
        for id in np.sort(np.unique(pid_list)):
            # current_features = features[pid_list==id, :]
            cam_center = []
            for cid in np.sort(np.unique(cid_list)):
                if not any(np.logical_and(pid_list==id,cid_list==cid)): continue
                cam_center.append(features[np.logical_and(pid_list==id,cid_list==cid), :].mean(axis=0))
            cam_var += np.concatenate(cam_center, axis=0).var(axis=0).sum()
        print("Camera variance :", cam_var)  

        
        # compute pose variance
        pose_var = 0
        for id in np.sort(np.unique(pid_list)):
            # current_features = features[pid_list==id, :]
            cam_center = []
            for poseid in np.sort(np.unique(poseid_list)):
                if not any(np.logical_and(pid_list==id,poseid_list==poseid)): continue
                cam_center.append(features[np.logical_and(pid_list==id,poseid_list==poseid), :].mean(axis=0))
            pose_var += np.concatenate(cam_center, axis=0).var(axis=0).sum()
        print("Pose variance :", pose_var)  

if __name__ == '__main__':
    # config
    weight_root = "G:/VCPAI_backup/newest/pose_logs/resnet50"

    
    np.random.seed(0)
    source, target = "market1501", "dukemtmc-reid"
    root = data_dir[target]
    run(source, target, root, weight_root)

    np.random.seed(0)
    source, target = target, source
    root = data_dir[target]
    run(source, target, root, weight_root)

    