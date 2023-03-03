# Person re-identification with pose and camera invariance Adversarial Neural Network

## Setup environment
Python version : 3.10.6
Ptorch version : 1.13.0
Torchvision version : 0.14.0 
CUDA : 11.7

Install code requirement
```
conda create -n pat python==3.10.6 -y
conda activate pat
pip --default-timeout=1000 install -r freeze_env/requirements.txt
```


## Running
Option
Dataset : "market1501", "dukemtmc-reid", "lab_data", "msmt17", "cuhk03"
Arch : "resnet50", "resnet_ibn50a", "osnet_ain_x0_5"

```bash
cd SSKD
bash scripts/pose_cam_at_mmt.sh ${SOURCE} ${TARGET} ${ARCH}
```


## Acknowledgments
Our code is based on [open-reid](https://github.com/Cysu/open-reid), [DomainAdaptiveReID](https://github.com/LcDog/DomainAdaptiveReID), [SSKD](https://github.com/PRIS-CV/SSKD) and [MMT](https://github.com/yxgeee/MMT),  if you use our code, please also cite their paper.

## Citation
If you find this code useful for your research, please cite our paper
```
 @phdthesis{Lin_2023, type={M.S. thesis}, title={Person re-identification with pose and camera invariance Adversarial Neural Network}, school={Graduate Institute of Automation Technology, National Taipei Univ. of Technology, Taipei}, author={Lin Bo-Yu}, year={2023}, language={zh-tw} }
```