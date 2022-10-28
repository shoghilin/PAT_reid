from __future__ import print_function, absolute_import
import os.path as osp
import glob, json
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}

class MSMT17(BaseImageDataset):
    # dataset_dir = 'msmt17/'
    dataset_dir = ''
    dataset_url = None

    def __init__(self, root='', verbose=True, wo_filter=False, **kwargs): 
        '''
        * wo_filter : set it true when you don't need filter out multiple pose samples.
        '''       
        super(MSMT17, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Dataset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )
        
        # read the pose info
        self.wo_filter = wo_filter
        pose_dir = osp.join(self.dataset_dir, main_dir, 'pose_labels_msmt17.json')
        with open(pose_dir, 'r') as f:
            self.pose = json.load(f)

        self._check_before_run()
        
        self.num_pose_cluster = 0
        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            # get the image path for current dataset
            img_path = f"{img_path.split('/')[0]}_c{camid+1}_0{img_path.split('_')[1]}.jpg"
            img_path = osp.join(dir_path, img_path)
            if dir_path == self.train_dir and not self.wo_filter:
                if osp.splitext(osp.basename(img_path))[0] not in self.pose.keys():
                    continue
                else:
                    poseid = self.pose[osp.splitext(osp.basename(img_path))[0]]
                    data.append((img_path, pid, camid, poseid))

                
                if self.num_pose_cluster-1 < poseid:
                    self.num_pose_cluster = poseid+1

            else:
                data.append((img_path, pid, camid, -1))

        return data

