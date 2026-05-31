#!/usr/bin/env bash
# Launch the reader server with CUDA libraries and GPU TTS provider on the path.
PYLIB=/home/zaeem/.pyenv/versions/3.10.14/lib/python3.10/site-packages

export LD_LIBRARY_PATH="\
$PYLIB/nvidia/cuda_runtime/lib:\
$PYLIB/nvidia/cublas/lib:\
$PYLIB/nvidia/cudnn/lib:\
$PYLIB/nvidia/curand/lib:\
$PYLIB/nvidia/cufft/lib:\
${LD_LIBRARY_PATH}"

export ONNX_PROVIDER=CUDAExecutionProvider

exec python "$(dirname "$0")/server.py"
