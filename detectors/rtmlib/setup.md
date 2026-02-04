# RTMLib setup

## Requirements
- Install requirements `pip install -r requirements.txt`

### If you want to use CPU
- `pip install onnxruntime`

### If you want to use GPU
- Automatic setup `pip install onnxruntime-gpu[cuda,cudnn]`
- If that fails, uninstall onnxruntime and perform a manual installation
  - Download and install [CUDA](https://developer.nvidia.com/cuda-downloads) and [CUDNN](https://developer.nvidia.com/cudnn-downloads) and make sure they're accessible through PATH
  - `pip install onnxruntime-gpu`
  - For more information use the [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

## Model and weights
- Download [YOLOX_x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip)
- Find `end2end.onnx` inside the archive, rename it as `yolox_x.onnx` and move it to `detectors/rtmlib/models`.
- Download [RTMPose-x](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip)
- Find `end2end.onnx` inside the archive, rename it as `rtmpose_x.onnx` and move it to `detectors/rtmlib/models`


You can get different models on the [RTMLib GitHub repository](https://github.com/Tau-J/rtmlib?tab=readme-ov-file#model-zoo).