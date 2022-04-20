import torch
from torch import nn

def create_cam_disc(args):
    if args.arch == "osnet_ain_x0_5":
        return OSNet_AIN_DISC(args.c_dim)
    elif args.arch == "resnet50":
        return ResNet50_DISC(args.c_dim)
    else:
        raise NameError("Cam discriminator not found!!!")

def create_pose_disc(args):
    if args.arch == "osnet_ain_x0_5":
        return OSNet_AIN_DISC(args.num_pose_cluster)
    elif args.arch == "resnet50":
        return ResNet50_DISC(args.num_pose_cluster)
    else:
        raise NameError("Cam discriminator not found!!!")


class ResNet50_DISC(nn.Module):
    def __init__(self, out_dim):
        super(ResNet50_DISC, self).__init__()
        self.model = nn.Sequential(*[
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        ])

    def forward(self, x):
        return self.model(x)

class OSNet_AIN_DISC(nn.Module):
    def __init__(self, out_dim):
        super(OSNet_AIN_DISC, self).__init__()
        self.model = nn.Sequential(*[
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        ])

    def forward(self, x):
        return self.model(x)