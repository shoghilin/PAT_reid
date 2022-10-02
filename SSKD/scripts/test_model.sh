#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
MODE=$4

CUDA=0
LOG_PATH="../pose_logs"
DATA_PATH="../Datasets"

CUDA_VISIBLE_DEVICES=${CUDA} \
python examples/test_model.py -dt ${TARGET} -a ${ARCH} \
    --resume ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-${MODE}/model_best.pth.tar \
    --logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-${MODE} \
    --data-dir ${DATA_PATH}  --visrank    # --rr-gpu 