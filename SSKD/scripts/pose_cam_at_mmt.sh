#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3

CUDA=0
LOG_PATH="../pose_logs"
# DATA_PATH="/home/lab314/HDD/shoghi/Datasets"
DATA_PATH="../Datasets"

# parameter
ITERS=500
WORKER=4

pretrain(){
	CUDA_VISIBLE_DEVICES=0 \
	python examples/pretrain.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} --seed ${SEED} --margin 0.0 \
		--num-instances 4 -b 32 -j ${WORKER} --warmup-step 10 --lr 0.00035 --milestones 40 70 --iters 100 --epochs 80 --eval-step 40 \
		--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-${SEED} --data-dir ${DATA_PATH}
}

mmt_pcat(){
	CUDA_VISIBLE_DEVICES=${CUDA} \
	python examples/pose_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
		--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
		--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
		--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
		--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
		--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pose-cam-mmt \
	    --data-dir ${DATA_PATH} --print-freq 50\
		--pose_reid_weight 0.3 --cam_reid_weight 0.3 -j ${WORKER} \
		--pose_mode ${POSE_MODE} --none_mode ${NONE_MODE} --num_pose_cluster ${NUM_POSE_CLUSTER} \
		# --rr-gpu 
}

mmt_pat(){
	CUDA_VISIBLE_DEVICES=${CUDA} \
	python examples/pose_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
		--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
		--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
		--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
		--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
		--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pose-mmt \
		--data-dir ${DATA_PATH} --print-freq 50 --wo_cat\
		--pose_reid_weight 0.3 -j ${WORKER}  \
		--pose_mode ${POSE_MODE} --none_mode ${NONE_MODE} --num_pose_cluster ${NUM_POSE_CLUSTER} \
		# --rr-gpu 
}

mmt_cat(){
	CUDA_VISIBLE_DEVICES=${CUDA} \
	python examples/pose_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
		--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
		--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
		--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
		--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
		--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-cam-mmt \
	    --data-dir ${DATA_PATH} --print-freq 50 --wo_pat\
		--cam_reid_weight 0.3 -j ${WORKER}  \
		# --rr-gpu 
}

mmt(){
	CUDA_VISIBLE_DEVICES=${CUDA} \
	python examples/pose_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
		--num-instances 8 --lr 0.00035 --iters ${ITERS} -b 64 --epochs 40 \
		--soft-ce-weight 0.5 --soft-tri-weight 0.8 --dropout 0 --lambda-value 0 \
		--init-1 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
		--init-2 ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
		--logs-dir ${LOG_PATH}/${SOURCE}TO${TARGET}/${ARCH}-mmt \
		--data-dir ${DATA_PATH} --print-freq 50 --wo_cat --wo_pat -j ${WORKER} \
		# --rr-gpu 
}

run(){
	echo "Training ${SOURCE} to ${TARGET}" >> "../pose_logs/training_time.txt"
	d1=$(date +"%s")

	if [ "$MODE" = "MMT" ]; then
		mmt
	elif [ "$MODE" =  "PAT" ]; then
		mmt_pat	
	elif [ "$MODE" =  "CAT" ]; then
		mmt_cat
	elif [ "$MODE" =  "PCAT" ]; then
		mmt_pcat
	fi

	d2=$(date +"%s")
	cost_time=$((d2-d1))
	echo "MMT+$MODE process time $(date -d@$cost_time -u +%H:%M:%S)" >> "../pose_logs/training_time.txt"
}

NUM_POSE_CLUSTER=8
NONE_MODE="new_label"
POSE_MODE="each_cam"

# Pretrain
SEED=1
pretrain
SEED=2
pretrain


SEED=2
# training
MODE="PAT"
run