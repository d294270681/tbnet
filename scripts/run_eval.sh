#!/bin/bash
if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: bash run_train.sh [CHECKPOINT_ID] [DATA_NAME](optional, default steam) [DEVICE_ID] (optional) [DEVICE_TARGET] (optional, default Ascend)
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

CHECKPOINT_ID=$1
DATA_NAME=$2
DEVICE_ID=$3
DEVICE_TARGET=$4

python ../eval.py --checkpoint_id $CHECKPOINT_ID --dataset $DATA_NAME --device_target $DEVICE_TARGET --device_id $DEVICE_ID