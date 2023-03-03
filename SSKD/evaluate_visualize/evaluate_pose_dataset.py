from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import pandas as pd
from PIL import Image
import sys, time
sys.path.append('..')

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset

from sskd import datasets
from sskd import models
from sskd.evaluators import Evaluator
from sskd.utils.data import transforms as T
from sskd.utils.data.preprocessor import Preprocessor
from sskd.utils.logging import Logger
from sskd.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict

class Pose_dataset(Dataset):
    def __init__(self, name, transform):
        name = 'market' if 'market' in name else 'duke'
        df_gallery = pd.read_csv(f"./{name}_pose_gallery.csv", index_col=0)
        df_gallery['fname'] = list(map(lambda x: osp.basename(x), df_gallery['fpath']))
        df_query = pd.read_csv(f"./{name}_pose_query.csv", index_col=0)
        df_query['fname'] = list(map(lambda x: osp.basename(x), df_query['fpath']))
        
        self.fpath_dict = {fname:fpath for fpath, _, _, _, fname in df_gallery.to_numpy()} | {fname:fpath for fpath, _, _, _, fname in df_query.to_numpy()}

        self.gallery = [(osp.basename(img_path), pid, camid, poseid) for _, pid, camid, poseid, img_path in df_gallery.to_numpy()]
        self.query = [(osp.basename(img_path), pid, camid, poseid) for _, pid, camid, poseid, img_path in df_query.to_numpy()]
        # print(self.gallery)
        # exit(0)
        self.dataset = list(set(self.query) | set(self.gallery))
        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        fname, pid, camid, poseid = self.dataset[indices]
        fpath = self.fpath_dict[fname]
        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid


def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)


    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
             T.ToTensor(),
             normalizer
         ])

    
    dataset = Pose_dataset(name, transform=test_transformer)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    # log_dir = osp.dirname(args.resume)
    # sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    # print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset_target, test_loader_target = \
        get_data(args.dataset_target, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)

    # Load from checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model)
    start_epoch = checkpoint['epoch']
    best_mAP = checkpoint['best_mAP']
    print("=> Checkpoint of epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    evaluator = Evaluator(model)
    print("Test on the target domain of {}:".format(args.dataset_target))
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True, rerank=args.rerank, visrank=args.visrank, args=args)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, required=True,
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # testing configs
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--visrank', action='store_true',
                        help="visualize ranking")
    parser.add_argument('--visrank_topk', type=int, default=10)
    parser.add_argument('--data_type', default="image")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
