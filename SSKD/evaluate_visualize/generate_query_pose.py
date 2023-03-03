# This code is for generate queryset with the largest pose variance
import os, re, json, sys
sys.path.append('..')
import os.path as osp
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

from evaluate_visualize.compute_variance import create_model_AND_extract_feature
from sskd.utils.logging import Logger
from sskd.utils.data import transforms as T
from sskd.utils.data.preprocessor import Preprocessor

dataINFOs = {
    "market":{
        "name":"market",
        "img_root":"G:/Github/Datasets/market1501/Market-1501-v15.09.15/bounding_box_test",
        "query_root":"G:/Github/Datasets/market1501/Market-1501-v15.09.15/query",
        "checkpoint_paths":"G:/VCPAI_backup/newest/pose_logs/resnet50/market1501TOdukemtmc-reid/resnet50-pretrain-1/model_best.pth.tar",
        "query_id_amount":750,
        "num_query_samples":3368
    },
    "duke":{
        "name":"duke",
        "img_root":"G:/Github/Datasets/dukemtmc-reid/DukeMTMC-reID/bounding_box_test",
        "query_root":"G:/Github/Datasets/dukemtmc-reid/DukeMTMC-reID/query",
        "checkpoint_paths":"G:/VCPAI_backup/newest/pose_logs/resnet50/dukemtmc-reidTOmarket1501/resnet50-pretrain-1/model_best.pth.tar",
        "query_id_amount":702,
        "num_query_samples":2228
    }

}

def create_dataloader(dataINFO):
    # load pose infomation...
    with open(f"G:/Code/pose_estimation/analysis/pose_labels/each_cam-new_label-8/pose_labels_{dataINFO['name']}_test.json", 'r') as f:
        info = json.load(f)

    df = pd.DataFrame.from_dict(info, orient='index', columns=['poseid'])
    pcid = pd.DataFrame(list(map(lambda x:list(map(int, re.findall("(-?\d*)_c(\d*)", x)[0])), df.index)), columns=['pid', 'camid'], index=df.index)
    df = pd.concat([df, pcid], axis=1)
    # remove noise sample (those samples that have pid equal to -1)
    df = df[(df['pid']!=-1) & (df['pid']!=0)]
    df['fpath'] = list(map(lambda x:osp.join(dataINFO['img_root'], f"{x}.jpg") , df.index))
    df = df[['fpath', 'pid', 'camid', 'poseid']]
    dataset = df.to_numpy()

    
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    transform = T.Compose([
                T.Resize((256, 128), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                normalizer
            ])
    test_loader = DataLoader(
            Preprocessor(dataset, root=dataINFO['img_root'], transform=transform, base_pose=True),
            batch_size=1, num_workers=8,
            shuffle=False, pin_memory=True)

    return df, test_loader


if __name__=='__main__':
    np.random.seed(1)
    
    dataName = 'market'
    dataINFO = dataINFOs[dataName]    
    
    sys.stdout = Logger(f'./pose_data_generate_log_{dataName}.txt')
    
    
    # create dataloader
    df, test_loader = create_dataloader(dataINFO=dataINFO)

    # create model, extract feature, and reduce dimention to 2d
    features, pid_list, cid_list, poseid_list = create_model_AND_extract_feature(dataINFO['checkpoint_paths'], test_loader)
        
    # compute pose variance
    pose_var = {}
    for id in np.sort(np.unique(df['pid'])):
        # current_features = features[pid_list==id, :]
        pose_center = []
        for poseid in np.sort(np.unique(poseid_list)):
            if not any(np.logical_and(pid_list==id,poseid_list==poseid)): continue
            pose_center.append(features[np.logical_and(pid_list==id,poseid_list==poseid), :].mean(axis=0))
        pose_var[id] = np.concatenate(pose_center, axis=0).var(axis=0).sum()
    
    # get the person who have the top k largest pose variance
    df_top_k = pd.DataFrame.from_dict(pose_var, orient='index', columns=['pose_var']).sort_values(by=['pose_var']).iloc[:int(len(np.unique(df['pid']))/2)]
    print(f"Selected top {df_top_k.shape[0]} identity that contain larger pose variance.")
    
    # query and gallery samples
    overall_filename = list(map(lambda x:osp.join(dataINFO['img_root'], x), os.listdir(dataINFO['img_root']))) \
                    + list(map(lambda x:osp.join(dataINFO['query_root'], x), os.listdir(dataINFO['query_root'])))
    overall_filename = [i for i in overall_filename if '.jpg' in i]
    df_overall = pd.DataFrame(list(map(lambda x:[x]+list(map(int, re.findall("(-?\d*)_c(\d*)", x)[0])), overall_filename)), columns=['fpath', 'pid', 'camid'])
    df_overall['poseid'] = -1
    df_overall = df_overall[df_overall['pid']!=-1]
    df_query = df_overall[df_overall['pid'].isin(df_top_k.index)].sample(n=int(dataINFO["num_query_samples"]/2), replace=False)
    df_gallery = df_overall[~df_overall['fpath'].isin(df_query['fpath'])]
    print("Generate new gallery and query sets.")
    print(f"Query\t| \tnum_id : {len(pd.unique(df_query['pid']))},\tnum_sample : {df_query.shape[0]}")
    print(f"Gallery\t| \tnum_id : {len(pd.unique(df_gallery['pid']))},\tnum_sample : {df_gallery.shape[0]}")
    
    # Save the generated result
    df_gallery.to_csv(f"{dataName}_pose_gallery.csv")
    df_query.to_csv(f"{dataName}_pose_query.csv")
    print(f"{dataName} pose query and gallery set generated.")
