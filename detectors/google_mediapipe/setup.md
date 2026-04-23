 # ⚙️ Google MediaPipe Setup

[🔗 **Official Documentation**](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

MediaPipe Pose Landmarker is a robust and lightweight solution by Google for keypoint extraction.

## 📦 1. Installation
 
 MediaPipe is already included in the main `requirements.txt`. If you need to install it manually, run:
 ```bash
 pip install mediapipe
 ```
 
 ## 🧠 2. Model Weights
 
 To use the detector, you must download the underlying task model:
 
 1. **Download the [`pose_landmarker_heavy.task` model](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task)**.
 2. Move the downloaded file directly into the `detectors/google_mediapipe/models/` directory.
 
 ## 💡 3. Notes
 
 You can explore and download different (lighter/faster) models from the [MediaPipe Model Zoo](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). If you do, make sure to update your application configuration YAML to match the new filename.