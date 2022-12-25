import os, re, sys
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

def create_dataset(root):
    fList = os.listdir(root)
    df = pd.DataFrame({"fname":fList, "pid":list(map(get_label, fList)), "cid":list(map(get_cam, fList)), "none":fList})
    selected_id = np.sort(np.random.choice(df['pid'].unique(), 20, replace=False))
    df = df[df.pid.isin(selected_id)]
    return df.to_numpy(), selected_id

def setupDataLoader(dataset, root):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    transform = T.Compose([
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                normalizer
            ])
    test_loader = DataLoader(
            Preprocessor(dataset, root=root, transform=transform),
            batch_size=200, num_workers=8,
            shuffle=False, pin_memory=True)
    return test_loader




def create_model_AND_extract_feature(checkpoint_path, test_loader):
    # build model
    model = models.create('resnet50', num_features=0, dropout=0, num_classes=1000).cuda()
    model = nn.DataParallel(model)
    initial_weights = load_checkpoint(checkpoint_path)
    copy_state_dict(initial_weights['state_dict'], model)
    model.eval()    

    # feature extraction
    features, pid_list = [], []
    with torch.no_grad():
        for imgs, _, pids, _ in test_loader:
            outputs = model(imgs.cuda())
            pid_list.append(pids)
            features.append(outputs)            
    features = torch.concat(features, dim=0).cpu().numpy()
    pid_list = torch.concat(pid_list, dim=0).cpu().numpy()

    # dimention reduction
    X_tsne = manifold.TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0).fit_transform(features)

    return features, pid_list, X_tsne

def plot_result(X_tsne, selected_id, features, pid_list):
    for id in selected_id:
        X_id = X_tsne[pid_list==id, :]
        plt.scatter(X_id[:, 0], X_id[:, 1], label=f"{str(id)}({features[pid_list==id, :].var(axis=0).sum():.2f})")
        plt.text(X_id[:, 0].mean(), X_id[:, 1].mean(), s=str(id))

    # Shrink current axis's height by 10% on the bottom
    box = plt.get_position()
    plt.set_position([box.x0, box.y0 + box.height * 0.2,
                    box.width, box.height * 0.9])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), title='class(variance)',
            fancybox=True, shadow=True, ncol=4)


def run(source, target, root, weight_root):
    print(f"Start process task {source}->{target}")
    # prepare data (create dataset & setup loader)
    dataset, selected_id = create_dataset(root)
    test_loader = setupDataLoader(dataset, root)    

    for mode in mode_types.keys():
        checkpoint_path=f"{weight_root}/{source}TO{target}/{mode_types[mode]}/model_best.pth.tar"

        # create model, extract feature, and reduce dimention to 2d
        features, pid_list, X_tsne = create_model_AND_extract_feature(checkpoint_path, test_loader)
        
        # plot figure
        fig, ax = plt.subplots(figsize=(8, 7))
        
        for id in selected_id:
            X_id = X_tsne[pid_list==id, :]
            ax.scatter(X_id[:, 0], X_id[:, 1], label=f"{str(id)}({features[pid_list==id, :].var(axis=0).sum():.2f})")
            ax.text(X_id[:, 0].mean(), X_id[:, 1].mean(), s=str(id))
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2,
                        box.width, box.height * 0.9])
        
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), title='class(variance)',
                fancybox=True, shadow=True, ncol=4)

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        fig.savefig(f"{source}2{target}_{mode}.png", pad_inches = 0.05, bbox_inches="tight")
        # sys.exit()


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

    