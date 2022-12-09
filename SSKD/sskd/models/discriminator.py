import torch
from torch import nn

def create_disc(args, mode):
    out_dim = {"pose":args.num_pose_cluster, "camera":args.c_dim}
    if "osnet" in args.arch:
        return OSNet_AIN_DISC(out_dim[mode])
    elif "resnet" in args.arch:
        return ResNet50_DISC(out_dim[mode])
    else:
        raise NameError(f"{mode} discriminator not found!!!")


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