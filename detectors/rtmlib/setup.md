 # ⚙️ RTMLib Setup
 
[🔗 **Official GitHub Repository**](https://github.com/Tau-J/rtmlib)

RTMLib provides highly optimized, state-of-the-art pose estimation models running on ONNX.

## 📦 1. Installation

The base requirements are handled by the main `requirements.txt`. However, you need to choose your ONNX Execution Provider (CPU vs GPU).
 
### Option A: CPU Only
```bash
pip install onnxruntime
```

### Option B: GPU (CUDA) Acceleration
 For automatic setup attempt, run:
 ```bash
 pip install onnxruntime-gpu[cuda,cudnn]
 ```
 *If the automatic setup fails, uninstall `onnxruntime-gpu` and perform a manual installation:*
 1. Download and install [CUDA](https://developer.nvidia.com/cuda-downloads) and [CUDNN](https://developer.nvidia.com/cudnn-downloads). Ensure they are properly added to your system's `PATH`.
 2. Run `pip install onnxruntime-gpu`.
 3. Refer to the [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for specific CUDA/cuDNN compatibility versions.
 
 ## 🧠 2. Model Weights
 
 RTMLib requires two separate ONNX models: a bounding box detector (YOLOX) and a pose estimator (RTMPose).
 
 1. **Download the Detector (YOLOX-x):**
    - **Download [YOLOX_x archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip)**.
    - Extract the archive, locate the `end2end.onnx` file inside.
    - Rename it to `yolox_x.onnx` and move it into the `detectors/rtmlib/models/` directory.
 
 2. **Download the Pose Estimator (RTMPose-x):**
    - **Download [RTMPose-x archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip)**.
    - Extract the archive, locate the `end2end.onnx` file inside.
    - Rename it to `rtmpose_x.onnx` and move it into the `detectors/rtmlib/models/` directory.
 
 ## 💡 3. Notes

 You can download different model variants (e.g., Nano, Small, Large) from the [RTMLib Model Zoo](https://github.com/Tau-J/rtmlib?tab=readme-ov-file#model-zoo). If you do, remember to update your `app_config.yaml` to match your new filenames.
