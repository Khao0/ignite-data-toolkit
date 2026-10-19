#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker run --rm -it \
    --device /dev/fuse \
    --cap-add SYS_ADMIN \
    --security-opt apparmor=unconfined \
    -v "$(pwd)":/ignite_data_toolkit \
    -w /ignite_data_toolkit/code/he_segmentation/ \
    --gpus all \
    ignite_data_toolkit:he
