 # ⚙️ RIFE Setup
 
 [🔗 **Official GitHub Repository**](https://github.com/hzwer/ECCV2022-RIFE)
Y
 RIFE (Real-Time Intermediate Flow Estimation) is used in this pipeline for smart, AI-driven video FPS interpolation.
 
 ## 🧠 Model Weights
 
 To use RIFE, you need to manually download the pre-trained flow model:
 
 1. Download the pre-trained model archive from [Google Drive](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view)
 2. Open the downloaded archive and extract its contents.
 3. Locate the `train_log` folder inside the extracted files.
 4. Move the `flownet.pkl` file into the `rife/train_log/` directory in this project.