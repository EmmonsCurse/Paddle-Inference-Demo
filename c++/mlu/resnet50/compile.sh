#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "${work_path}/../../lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/lib"
  exit 1
fi

# 2. check CUSTOM_DEVICE_ROOT env
if [ -z "$CUSTOM_DEVICE_ROOT" ]; then
  echo "Error: CUSTOM_DEVICE_ROOT is not set or is empty"
  exit 1
fi

# 如果 CUSTOM_DEVICE_ROOT 不为空，继续执行脚本
echo "CUSTOM_DEVICE_ROOT is set to '$CUSTOM_DEVICE_ROOT'"

# 3. check CMakeLists exists
if [ ! -f "${work_path}/CMakeLists.txt" ]; then
  cp -a "${work_path}/../../lib/CMakeLists.txt" "${work_path}/"
fi

# 4. compile
mkdir -p build
cd build
rm -rf *

# same with the resnet50_test.cc
DEMO_NAME=resnet50_test

WITH_MKL=ON
WITH_ARM=OFF

LIB_DIR=${work_path}/../../lib/paddle_inference

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_ARM=${WITH_ARM} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_STATIC_LIB=OFF 

if [ "$WITH_ARM" == "ON" ];then
  make TARGET=ARMV8 -j
else
  make -j
fi
