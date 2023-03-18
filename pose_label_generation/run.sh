NUM_CLUSTER=8
MODE="each_cam"  # "each_cam", "overall"
NONE_MODE="new_label" # ignore, cam_labels, new_label
DATASET="lab_combine" # market, duke, msmt17, lab_combine

function run_all(){
    # python clustering.py --dataset market --n_cluster ${NUM_CLUSTER} --mode ${MODE} --none_mode ${NONE_MODE} --save_json
    # python clustering.py --dataset market_test --n_cluster ${NUM_CLUSTER} --mode ${MODE} --none_mode ${NONE_MODE} --save_json
    # python clustering.py --dataset duke --n_cluster ${NUM_CLUSTER} --mode ${MODE} --none_mode ${NONE_MODE} --save_json
    python clustering.py --dataset duke_test --n_cluster ${NUM_CLUSTER} --mode ${MODE} --none_mode ${NONE_MODE} --save_json
    # python clustering.py --dataset msmt17 --n_cluster ${NUM_CLUSTER} --mode ${MODE} --none_mode ${NONE_MODE} --save_json
    # python clustering.py --dataset lab_combine --n_cluster ${NUM_CLUSTER} --mode ${MODE} --none_mode ${NONE_MODE} --save_json
}


NUM_CLUSTER=8
MODE="each_cam"  # "each_cam", "overall"
NONE_MODE="new_label" # ignore, cam_labels, new_label
run_all
# NONE_MODE="cam_labels" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="ignore" # ignore, cam_labels, new_label
# run_all


# MODE="overall"  # "each_cam", "overall"
# NONE_MODE="ignore" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="cam_labels" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="new_label" # ignore, cam_labels, new_label
# run_all


# NUM_CLUSTER=4
# MODE="each_cam"  # "each_cam", "overall"
# NONE_MODE="cam_labels" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="ignore" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="new_label" # ignore, cam_labels, new_label
# run_all


# MODE="overall"  # "each_cam", "overall"
# NONE_MODE="ignore" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="cam_labels" # ignore, cam_labels, new_label
# run_all
# NONE_MODE="new_label" # ignore, cam_labels, new_label
# run_all