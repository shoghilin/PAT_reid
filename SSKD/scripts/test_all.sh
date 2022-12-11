#!/bin/sh
ARCH="resnet50"
MODE="cam-mmt"

CUDA=0
LOG_PATH="G:/VCPAI_backup/newest/pose_logs/"
DATA_PATH="../../Datasets"

evaluate(){	
    CUDA_VISIBLE_DEVICES=${CUDA} \
    python examples/test_model.py -dt ${TARGET} -a ${ARCH} \
        --resume ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-${MODE}/model_best.pth.tar \
        --logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-${MODE} \
        --data-dir ${DATA_PATH}  --visrank    # --rr-gpu 
}

# LOG_PATH="G:/VCPAI_backup/newest/pose_logs/resnet_ibn/pose_logs"
# ARCH="resnet_ibn50a"
LOG_PATH="G:/VCPAI_backup/newest/pose_logs/osnet_ain"
ARCH="osnet_ain_x0_5"

SOURCE=dukemtmc-reid
TARGET=market1501
for i in 'mmt' 'cam-mmt' 'pose-mmt';
do
    MODE=${i}
    evaluate
done

SOURCE=market1501
TARGET=dukemtmc-reid
for i in 'mmt' 'cam-mmt' 'pose-mmt';
do
    MODE=${i}
    evaluate
done

SOURCE=dukemtmc-reid
TARGET=msmt17
for i in 'mmt' 'cam-mmt' 'pose-mmt';
do
    MODE=${i}
    evaluate
done

SOURCE=market1501
TARGET=msmt17
for i in 'mmt' 'cam-mmt' 'pose-mmt';
do
    MODE=${i}
    evaluate
done
