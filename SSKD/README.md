# Enhencing pose and camera invariance for UDA person re-identification

# Requirements
python 3.6

PyTorch >= 1.1.1

torchvision >= 0.2.1

# Running
```
bash scripts/pose_cam_at_mmt.sh $SOURCE $TARGET $ARCH $mode
```
Option
Dataset : "market1501", "dukemtmc-reid", "lab_data"
Arch : "resnet50", "resnet_ibn50a", "osnet_ain_x0_5"

Step 1: Train on source dataset
```Swift
bash scripts/pose_cam_at_mmt.sh market1501 dukemtmc-reid resnet50
```

Step 2:Train on target dataset
```Swift
bash scripts/sskd_train.sh
```
