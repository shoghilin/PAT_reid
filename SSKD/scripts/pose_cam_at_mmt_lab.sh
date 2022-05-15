#!/bin/sh
conda activate ADCluster

SOURCE=market1501
# SOURCE=dukemtmc-reid
TARGET=lab_data

CUDA=4,5
SEED=3
# ARCH=resnet50
# ARCH=resnet_ibn50a
ARCH=osnet_ain_x0_5 
# ARCH=osnet_ain_x1_0
LOG_PATH="../pose_logs"
DATA_PATH="/home/lab314/HDD/shoghi/Datasets"

# parameter
ITERS=500

# Pretrain
# CUDA_VISIBLE_DEVICES=4 \
# python examples/pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --seed ${SEED} --margin 0.0 \
# 	--num-instances 4 -b 32 -j 4 --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 100 --epochs 80 --eval-step 40 \
# 	--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-${SEED} --data-dir ${DATA_PATH}

# Standard
# CUDA_VISIBLE_DEVICES=${CUDA} \
# python examples/pose_train_lab.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
# 	--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
# 	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
# 	--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
# 	--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
# 	--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pose-cam-mmt \
#     --data-dir ${DATA_PATH} --print-freq 50\
# 	--pose_reid_weight 0.3 --cam_reid_weight 0.3\
# 	# --rr-gpu 

# # MMT + PAT
CUDA_VISIBLE_DEVICES=${CUDA} \
python examples/pose_train_lab.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
	--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
	--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
	--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
	--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pose-mmt \
    --data-dir ${DATA_PATH} --print-freq 50 --wo_cat\
	--pose_reid_weight 0.3 \
	# --rr-gpu 

# MMT + CAT
# CUDA_VISIBLE_DEVICES=${CUDA} \
# python examples/pose_train_lab.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
# 	--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
# 	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
# 	--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
# 	--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
# 	--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-cam-mmt \
#     --data-dir ${DATA_PATH} --print-freq 50 --wo_pat\
# 	--cam_reid_weight 0.1 \
# 	# --rr-gpu 

# MMT
# CUDA_VISIBLE_DEVICES=${CUDA} \
# python examples/pose_train_lab.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
# 	--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
# 	--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
# 	--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
# 	--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
# 	--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-mmt \
#     --data-dir ${DATA_PATH} --print-freq 50 --wo_cat --wo_pat\
# 	# --rr-gpu 