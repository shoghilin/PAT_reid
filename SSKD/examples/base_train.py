from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys, time
sys.path.append('.')
import collections

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sskd import datasets
from sskd import models
from sskd.trainers import ClusterBaseTrainer, ATClusterBaseTrainer
from sskd.evaluators import Evaluator, extract_features
from sskd.utils.data import IterLoader
from sskd.utils.data import transforms as T
from sskd.utils.data.sampler import RandomMultipleGallerySampler
from sskd.utils.data.preprocessor import Preprocessor
from sskd.utils.logging import Logger
from sskd.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from sskd.utils.rerank import compute_jaccard_dist


start_epoch = best_mAP = 0

def get_data(name, data_dir, pose_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, pose_dir=pose_dir)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters, args, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=False, base_pose=not (args.wo_pat and args.wo_cat)),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args, classes):
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=classes)

    model.cuda()
    model = nn.DataParallel(model)

    initial_weights = load_checkpoint(args.init)
    copy_state_dict(initial_weights['state_dict'], model)

    return model

def create_disc(args):
    cam_disc = models.create_disc(args, mode='camera')
    pose_disc = models.create_disc(args, mode='pose')
    cam_disc = nn.DataParallel(cam_disc)
    pose_disc = nn.DataParallel(pose_disc)
    return cam_disc.to(args.device), pose_disc.to(args.device) 

def main():
    args = parser.parse_args()

    args.pose_dir = osp.join(args.data_dir, "pose_labels",
                                f"{args.pose_mode}-{args.none_mode}-{args.num_pose_cluster}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = False    

    log_name = 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_source = get_data(args.dataset_source, args.data_dir, args.pose_dir)
    dataset_target = get_data(args.dataset_target, args.data_dir, args.pose_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    tar_cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)
    sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers, testset=dataset_source.train)

    # config
    args.num_pose_cluster = dataset_target.num_pose_cluster
    args.c_dim = dataset_target.num_train_cams

    # Create model
    model = create_model(args, len(dataset_target.train))
    cam_disc, pose_disc = create_disc(args)

    # Evaluator
    evaluator = Evaluator(model)

    for epoch in range(args.epochs):
        dict_f, _ = extract_features(model, tar_cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f.values()))

        if (args.lambda_value>0):
            dict_f, _ = extract_features(model, sour_cluster_loader, print_freq=50)
            cf_s = torch.stack(list(dict_f.values()))
            rerank_dist = compute_jaccard_dist(cf, lambda_value=args.lambda_value, source_features=cf_s, use_gpu=args.rr_gpu).numpy()
        else:
            rerank_dist = compute_jaccard_dist(cf, use_gpu=args.rr_gpu).numpy()

        if (epoch==0):
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1) # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)] # tri_mat.dim=1
            tri_mat = np.sort(tri_mat,axis=None)
            rho = 1.6e-3 if args.dataset_target != 'msmt17' else 7e-4
            top_num = np.round(rho*tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps for cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - (1 if -1 in labels else 0)
        args.num_clusters = num_ids
        print('\n Clustered into {} classes \n'.format(args.num_clusters))

        # generate new dataset and calculate cluster centers
        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        for i, ((fname, _, cid, poseid), label) in enumerate(zip(dataset_target.train, labels)):
            if label==-1: continue
            new_dataset.append((fname,label,cid,poseid))
            cluster_centers[label].append(cf[i])

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
        model.module.classifier.weight.data[:args.num_clusters].copy_(F.normalize(cluster_centers, dim=1).float().cuda())

        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters, args, trainset=new_dataset)

        # Optimizer
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)
        disc_optimizer = [torch.optim.Adam(cam_disc.parameters(), lr=0.005), torch.optim.Adam(pose_disc.parameters(), lr=0.005)]

        # Trainer
        if args.wo_cat and args.wo_pat:
            trainer = ClusterBaseTrainer(model, num_cluster=args.num_clusters)

            train_loader_target.new_epoch()
            
            trainer.train(epoch, train_loader_target, optimizer, print_freq=args.print_freq, train_iters=len(train_loader_target))
        else:
            trainer = ATClusterBaseTrainer(model, cam_disc, pose_disc, args, num_cluster=args.num_clusters)

            train_loader_target.new_epoch()

            trainer.train(epoch, train_loader_target, optimizer, disc_optimizer,
                        print_freq=args.print_freq, train_iters=len(train_loader_target))

        def save_model(model, is_best, best_mAP):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP>best_mAP) 
            best_mAP = max(mAP, best_mAP)
            save_model(model, is_best, best_mAP)

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster Baseline Training")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help="use gpu or cpu to train model")
    ## data - pose label
    parser.add_argument('--pose_mode', default="each_cam", choices=["each_cam", "overall"],
                        help="Clustering based on overall dataset or each camera.")
    parser.add_argument('--none_mode', default="new_label", choices=["ignore", "cam_labels", "new_label"],
                        help="Different way of dealing the samples which did not detect pose.")    
    parser.add_argument('--num_pose_cluster', default='8', choices=['4', '8'],
                        help="The number of pose cluster for overall dataset of each camera.")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--pose_reid_weight', type=float, default=0.8)  # advarsarial training pose weight
    parser.add_argument('--cam_reid_weight', type=float, default=0.8)  # advarsarial training pose weight
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--lambda-value', type=float, default=0)
    parser.add_argument('--rr-gpu', action='store_true', 
                        help="use GPU for accelerating clustering")
    parser.add_argument('--wo_cat', action='store_true', 
                        help="Without camera discriminator.")
    parser.add_argument('--wo_pat', action='store_true', 
                        help="Without pose discriminator.")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    
    # counting training time
    start_time = time.time()
    main()
    training_time = time.gmtime(time.time()-start_time)
    print("Training time : {}H{}M{}S".format(
        training_time.tm_hour,
        training_time.tm_min,
        training_time.tm_sec
    ))