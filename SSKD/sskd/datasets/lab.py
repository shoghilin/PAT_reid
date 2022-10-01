from __future__ import print_function, absolute_import
import os, json
import os.path as osp

from ..utils.data import BaseImageDataset


class LAB(BaseImageDataset):
    """
    LAB collected dataset
    """

    def __init__(self, root, verbose=True, **kwargs):
        super(LAB, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        pose_dir = osp.join(self.dataset_dir, 'train/pose_labels_lab.json')
        with open(pose_dir, 'r') as f:
            self.pose = json.load(f)

        self._check_before_run()

        self.num_pose_cluster = 0
        train = self._process_dir()
        query = self._process_dir_test(self.query_dir)
        gallery = self._process_dir_test(self.gallery_dir)

        if verbose:
            print("=> LAB DATA loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self):
        dataset = []
        for pth in os.listdir(self.train_dir):
            img_path = osp.join(self.train_dir, pth)
            camid = pth[7]
            if osp.splitext(osp.basename(img_path))[0] not in self.pose.keys():
                continue
            else:
                poseid = self.pose[osp.splitext(osp.basename(img_path))[0]]
                dataset.append((img_path, 0, int(camid), poseid))

                
            if self.num_pose_cluster-1 < poseid:
                self.num_pose_cluster = poseid+1
        # print(dataset)

        return dataset

    def _process_dir_test(self, dir):
        dataset = []
        for pth in os.listdir(dir):
            img_path = osp.join(dir, pth)
            info = pth.split('_')
            pid, camid = int(info[1]), int(info[0][1])
            dataset.append((img_path, pid, camid, 0))
        
        return dataset