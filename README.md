# Person re-identification with pose and camera invariance Adversarial Neural Network

# Setup environment
Python version : 3.10.6
Ptorch version : 1.13.0
Torchvision version : 0.14.0 
CUDA : 11.7

Install requirement
```
conda create -n pat python==3.10.6 -y
conda activate pat
pip --default-timeout=1000 install -r freeze_env/requirements.txt
```


# Running
```
bash scripts/pose_cam_at_mmt.sh $SOURCE $TARGET $ARCH $mode
```
Option
Dataset : "market1501", "dukemtmc-reid", "lab_data", "msmt17", "cuhk03"
Arch : "resnet50", "resnet_ibn50a", "osnet_ain_x0_5"

```Swift
bash scripts/pose_cam_at_mmt.sh ${SOURCE} ${TARGET} ${ARCH}
```