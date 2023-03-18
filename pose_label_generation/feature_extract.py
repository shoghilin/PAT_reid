import torch
import os.path as osp
from PIL import Image
import torchvision.transforms as T
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from utils import read_json, process

class Pose(Dataset):
    def __init__(self, transform=None, dataset="market"):
        self.pose = read_json(f"pose_result/{dataset}_jsons")
        self.im_urls = [osp.join(f"pose_result/{dataset}_images", k+"_rendered.png") for k in self.pose.keys() if k[:2] != '-1' or k[:4] != '0000']
        self.transform = transform

    def __len__(self):
        return len(self.im_urls)

    def __getitem__(self, idx):
        im = Image.open(self.im_urls[idx])
        fname = list(self.pose.keys())[idx]

        if self.transform != None:
            im = self.transform(im)

        return im, fname

def get_dataloader(name="market"):
    # get dataset and data loader
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform = T.Compose([T.Resize((256, 128)), T.ToTensor(), normalize])
    data = Pose(transform=transform, dataset=name)
    dataloader = DataLoader(data, batch_size=128, shuffle=False)
    return dataloader



def feature_extraction(dataloader):
    # build model
    model = models.vgg19(pretrained=True).cuda()
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])

    # feature extraction
    fnames, fmaps = [], []
    with torch.no_grad():
        for i, (im, fname) in enumerate(dataloader):
            print("\r{}/{}".format(i+1, len(dataloader)), end=' ')
            output = model(im.cuda())
            fmaps.append(output)
            fnames = fnames + list(fname)

    fmaps = torch.cat(fmaps, dim=0).cpu().numpy()
    return fmaps, fnames

if __name__ == '__main__':
    dataloader = get_dataloader(name="market")
    fmaps, fnames = feature_extraction(dataloader=dataloader)
    print(fmaps.shape, len(fnames))