#!/bin/bash
if [[ $# -gt 3 ]]; then
    echo "Usage: bash run_train.sh [DATA_NAME](optional, default steam) [DEVICE_ID] (optional, default 0) [DEVICE_TARGET] (optional, default Ascend)
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

DATA_NAME=$1
DEVICE_ID=$2
DEVICE_TARGET=$3

python ../preprocess_dataset.py --dataset $DATA_NAME --device_target $DEVICE_TARGET
python ../train.py --dataset $DATA_NAME --device_target $DEVICE_TARGET --device_id $DEVICE_ID