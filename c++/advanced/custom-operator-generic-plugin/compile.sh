#!/bin/bash
set +x
set -e

work_path=$(dirname $(readlink -f $0))

# 1. check paddle_inference exists
if [ ! -d "${work_path}/../../lib/paddle_inference" ]; then
  echo "Please download paddle_inference lib and move it in Paddle-Inference-Demo/c++/lib"
  exit 1
fi

# 2. check CMakeLists exists
if [ ! -f "${work_path}/CMakeLists.txt" ]; then
  cp -a "${work_path}/../../lib/CMakeLists.txt" "${work_path}/"
fi

# 3. compile
mkdir -p build
cd build
rm -rf *

DEMO_NAME=custom_gap_op_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON
WITH_ONNXRUNTIME=OFF
WITH_SHARED_PHI=ON

LIB_DIR=${work_path}/../../lib/paddle_inference
CUDNN_LIB=/usr/lib/x86_64-linux-gnu/
CUDA_LIB=/usr/local/cuda/lib64
TENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6
CUSTOM_OPERATOR_FILES="custom_gap_op.cc;custom_gap_op.cu"


cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT} \
  -DCUSTOM_OPERATOR_FILES=${CUSTOM_OPERATOR_FILES} \
  -DWITH_ONNXRUNTIME=${WITH_ONNXRUNTIME} \
  -DWITH_SHARED_PHI=${WITH_SHARED_PHI}

make -j
